
import os
from argparse import ArgumentParser

import open3d as o3d
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from games.block_mesh_splatting.scene.two_gaussian_model import TwoGaussianModel
from renderer.gaussian_renderer_2d import render_part
from scene import Scene
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.render_utils import generate_path, create_videos, get_circle_traj

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_2d")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[100] * 8)
    parser.add_argument('--superquadric', '-su', action='store_true', default=False)
    parser.add_argument("--ratio_block_scene", default=0.25, type=float, help='ratio_block_scene')
    parser.add_argument("--scale_block_min", default=0.2, type=float, help='scale_block_min')
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    dataset.gs_type = args.gs_type
    dataset.ratio_block_scene = args.ratio_block_scene
    dataset.num_splats = args.num_splats
    dataset.scale_block_min = args.scale_block_min
    dataset.superquadric = args.superquadric
    dataset.img_skip_step = args.img_skip_step
    gaussians = TwoGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, None, gaussians, load_iteration=iteration, shuffle=False)
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_part, pipe, bg_color=bg_color)

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)

    if args.render_path:
        print("render videos ...")
        import copy
        import numpy as np
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 120
        cam_traj = generate_path(scene.getTestCameras(), n_frames=n_fames)
        cam = copy.deepcopy(scene.getTestCameras()[24])
        R_traj, T_traj = [t for t in get_circle_traj(N_views=n_fames)]
        w2c_ref = np.asarray((cam.world_view_transform).cpu().numpy())
        w2c_ref = w2c_ref @ np.diag([-1, -1, 1, 1])

        # c2w_ref = np.array(np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())))
        cam_traj = []
        for R_ in R_traj:
            cam = copy.deepcopy(scene.getTestCameras()[24])
            w2c = copy.deepcopy(w2c_ref) #@ np.diag([1, -1, -1, 1])
            w2c[:3, :3] = w2c[:3, :3] @ R_.numpy()
            w2c = w2c @ np.diag([-1, -1, 1, 1])

            cam.image_height = int(cam.image_height / 2) * 2
            cam.image_width = int(cam.image_width / 2) * 2

            cam.world_view_transform = torch.from_numpy(w2c).float().cuda()
            print(w2c)
            cam.full_proj_transform = (
                cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
            cam.camera_center = cam.world_view_transform.inverse()[3, :3]
            cam_traj.append(cam)

        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                      input_dir=traj_dir,
                      out_name='render_traj',
                      num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc,
                                                       depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))


