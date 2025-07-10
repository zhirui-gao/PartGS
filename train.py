
import os
import random
from random import randint
import torch
from utils.render_utils import save_img_u8
random.seed(42)
from utils.loss_utils import l1_loss, ssim
from renderer.gaussian_renderer import render
import sys
from scene import Scene
from games import (
    optimizationParamTypeCallbacks,
    gaussianModel
)
from pytorch3d.structures.meshes import join_meshes_as_scene
from utils import dtu_eval,blender_eval
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from pytorch3d.structures import Meshes
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from utils.plot import get_fancy_color


@torch.no_grad()
def reconstruction(cameras, gaussians, pipe, bg, train_dir):
    os.makedirs(train_dir, exist_ok=True)
    for idx, viewpoint_cam in tqdm(enumerate(cameras), desc="rendering training images"):
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        gt = viewpoint_cam.original_image[0:3, :, :]
        rgb = render_pkg['render']
        render_path = os.path.join(train_dir, "renders")
        gts_path = os.path.join(train_dir, "gt")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        save_img_u8(gt.permute(1, 2, 0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        save_img_u8(rgb.permute(1, 2, 0).cpu().numpy(),
                    os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def training(gs_type, data_type, dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, save_xyz):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = gaussianModel[gs_type](dataset.sh_degree, opt.ratio_block_scene, opt.scale_block_min)
    dataset.gs_type, dataset.data_type, opt.data_type = gs_type, data_type, data_type
    scene = Scene(dataset, opt, gaussians)
    gaussians.training_setup(opt)
    if hasattr(gaussians, 'prepare_scaling_rot'):
        gaussians.prepare_scaling_rot()

    if checkpoint:
        (model_params, first_iter) = torch.load(os.path.join(scene.model_path, "chkpnt" + str(checkpoint) + ".pth"))
        gaussians.restore_block(model_params, opt)

    bg_color = [1., 1., 1.] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        os.makedirs(f"{scene.model_path}/xyz", exist_ok=True)
        if save_xyz and (iteration % 5000 == 1 or iteration == opt.iterations):
            torch.save(gaussians.get_xyz, f"{scene.model_path}/xyz/{iteration}.pt")

        iter_start.record()

        gaussians.update_learning_rate(iteration, opt)
    

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
     
        gt_image = viewpoint_cam.original_image.cuda()
        
        use_reg = iteration < opt.coarse_learning_until_iter
        Ll1, loss, losses_reg = gaussians.summarize_losses(image, gt_image, viewpoint_cam, opt, use_reg)
    
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                losses_reg_formatted = {key: f"{value:.4f}" for key, value in losses_reg.items()}
                combined_losses = {**{"l1": f"{Ll1:.4f}"}, **losses_reg_formatted}
                progress_bar.set_postfix(combined_losses)
       
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background), losses_reg)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        
            gaussians.filter_primitive()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


            if (iteration in checkpoint_iterations) or iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture_block(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
              
            if iteration == opt.iterations:
                # save training and testing images:
                save_dir = os.path.join(scene.model_path, 'train', "block_ours_{}".format(opt.iterations))
                os.makedirs(save_dir, exist_ok=True)
                reconstruction(scene.getTrainCameras(), gaussians, pipe, background, save_dir)
                save_dir = os.path.join(scene.model_path, 'test', "block_ours_{}".format(opt.iterations))
                os.makedirs(save_dir, exist_ok=True)
                reconstruction(scene.getTestCameras(), gaussians, pipe, background, save_dir)

             
                if hasattr(scene, 'points_gt') and scene.points_gt is not None:
                    scale = torch.from_numpy(scene.scale_mat).to("cuda")
                    meshes = scene.gaussians.get_meshes()
                    scene_meshes = join_meshes_as_scene(meshes)
                    verts, faces = scene_meshes.get_mesh_verts_faces(0)
                    scene_meshes = Meshes((verts @ scale[:3, :3] + scale[:3, 3])[None], faces[None])
                    if opt.data_type == 'dtu':
                        scan_id = int(os.path.basename(scene.data_dir).replace('scan', ''))
                        dtu_eval.evaluate_mesh(scene_meshes, scan_id, os.path.dirname(scene.data_dir),
                                                scene.model_path, step=str(opt.iterations), save_viz=False,
                                                num_mesh=len(meshes))
                    else:
                        blender_eval.evaluate_mesh(scene_meshes, scene.points_gt,
                                                    scene.model_path, step=str(opt.iterations), num_mesh=len(meshes))

        if opt.mesh_from_iter < iteration <= opt.add_primitive_until_iter and \
                iteration % opt.add_primitive_interval == 0:
            gaussians.add_primitive(opt.data_type)


        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()


def training_refinement(gs_type, data_type, dataset, opt, opt_part, pipe, testing_iterations, saving_iterations, checkpoint_iterations,
                  checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # second stage of the model
    gaussians = gaussianModel[gs_type](dataset.sh_degree, args.ratio_block_scene, args.scale_block_min)

    dataset.gs_type, dataset.data_type = gs_type, data_type
    scene = Scene(dataset, opt_part, gaussians)
    (model_args, _) = torch.load(os.path.join(scene.model_path, f"chkpnt{str(opt.iterations)}.pth"))


    gaussians.create_from_block(model_args)

    gaussians.training_setup(opt_part)
    if checkpoint:
        (model_params, first_iter) = torch.load(
            os.path.join(scene.model_path, "part_chkpnt" + str(checkpoint) + ".pth"))
        gaussians.restore(model_params, opt)

    bg_color = [1., 1., 1.] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt_part.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt_part.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"],render_pkg["rend_alpha"],\
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"],

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

      
        gt_image_mask = viewpoint_cam.original_mask.cuda()
        opacity = opacity.clamp(1e-6, 1 - 1e-6).squeeze(0)
        loss_mask_entropy = -(gt_image_mask * torch.log(opacity) + (1-gt_image_mask) * torch.log(1 - opacity)).mean()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy
        loss_inter_reg = gaussians.inter_part_loss()
        loss = loss + loss_inter_reg

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss scale regularization
        if data_type == 'blender':
            scale_reg_loss = torch.clamp(gaussians.get_scaling - 0.005, min=0).sum()
            total_loss = loss + dist_loss + normal_loss + scale_reg_loss
        else:
            total_loss = loss + dist_loss + normal_loss

        # loss
        total_loss.backward()
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt_part.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations) or iteration == opt_part.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification

            if iteration <= opt_part.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt_part.densify_from_iter and iteration % opt_part.densification_interval == 0:
                    size_threshold = 20 if iteration > opt_part.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_part.densify_grad_threshold, opt_part.opacity_cull, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt_part.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt_part.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt_part.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations) or iteration == opt_part.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/part_chkpnt" + str(iteration) + ".pth")
                # gaussians.draw_gaussians_disk(scene.model_path, iteration)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, losses_reg=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        if losses_reg is not None:
            for loss_name, loss_item in losses_reg.items():
                tb_writer.add_scalar('train_loss_patches/' + loss_name, loss_item.item(), iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_dict = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_dict["render"], 0.0, 1.0)
                    if "acc" in render_dict:
                        acc = torch.clamp(render_dict["acc"], 0.0, 1.0)
                    if "opacity" in render_dict:
                        opacity = torch.clamp(render_dict["opacity"], 0.0, 1.0)
                    if "render_semantic" in render_dict:
                        part = torch.clamp(render_dict["render_semantic"], 0.0, 1.0)  #[C, H, W]
                        colors_define = get_fancy_color(part.shape[0]+1)
                        background = torch.sum(part, dim= 0)
                        predicted_classes = torch.argmax(part, dim=0)
                        color_image_part = torch.zeros((predicted_classes.shape[0], predicted_classes.shape[1], 3),
                                                       dtype=torch.float32)
                        for cls in range(part.shape[0]):
                            color_image_part[predicted_classes == cls] = colors_define[cls]
                        color_image_part[background<1e-1, :] = 0.
                        color_image_part = color_image_part.permute(2, 0, 1)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        from utils.general_utils import colormap
                        depth = render_dict["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        try:
                            rend_alpha = render_dict['rend_alpha']
                            rend_normal = render_dict["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_dict["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_dict["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass


                        if "acc" in render_dict:
                            tb_writer.add_images(config['name'] + "_view_{}/mask".format(viewpoint.image_name),
                                                 acc[None, None], global_step=iteration)

                        if "opacity" in render_dict:
                            tb_writer.add_images(config['name'] + "_view_{}/opacity".format(viewpoint.image_name),
                                                 opacity[None], global_step=iteration)
                        if "render_semantic" in render_dict:
                            tb_writer.add_images(config['name'] + "_view_{}/part".format(viewpoint.image_name),
                                                 color_image_part[None], global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_mask".format(viewpoint.image_name),
                                                 gt_mask[None,None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--data_type', type=str, default="dtu")
    parser.add_argument("--training_type",type=str, choices=["block", "part", "all"],  default="all", help="Training type: 'block' or 'part'")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[25000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[25000, 30000])
    parser.add_argument("--eval_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--start_checkpoint", '-load', type=str, default=None)
    parser.add_argument("--start_checkpoint_2dgs", type=str, default=None)
    parser.add_argument("--save_xyz", action='store_true')
    lp = ModelParams(parser)
    args, _ = parser.parse_known_args(sys.argv[1:])
    
    op = optimizationParamTypeCallbacks['gs_block'](parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)

    print("torch cuda: ", torch.cuda.is_available())
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.training_type == "block" or args.training_type == "all":
        training(
            'gs_block',
            args.data_type,
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint, args.debug_from, args.save_xyz
        )
        torch.cuda.empty_cache()

    if args.training_type == "part" or args.training_type == "all":
        parser_part = ArgumentParser(description="Training part script parameters")
        op_part = optimizationParamTypeCallbacks["gs_point"](parser_part)

        training_refinement(
            'gs_point',
            args.data_type,
            lp.extract(args), op.extract(args), op_part, pp.extract(args),
            args.test_iterations, args.save_iterations, args.checkpoint_iterations,
            args.start_checkpoint_2dgs, args.debug_from
        )

    print("\n Block-level training complete.")


