
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, opt: OptimizationParams=None, gaussians : GaussianModel=None, 
                 load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.data_dir = args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_block":
                print("Found sparse dirction, assuming colmap data set!")
                if args.data_type == "dtu":
                    scene_info = sceneLoadTypeCallbacks["Colmap_Block"](
                        args.source_path, args.images, args.eval, opt.num_splats,
                        opt.ratio_block_scene, opt.scale_block_min, args.img_skip_step,
                        render_camera = "cameras.npz"
                    )
                    self.gaussians.blocks = scene_info.blocks
                    self.gaussians.points = scene_info.points
                    self.points_gt = scene_info.points_gt
                    self.scale_mat = scene_info.scale_mat
                elif args.data_type == "replica":
                    scene_info = sceneLoadTypeCallbacks["Replica_Block"](
                        args.source_path, args.images, args.eval, opt.num_splats,
                        opt.ratio_block_scene, opt.scale_block_min
                    )
                    self.gaussians.blocks = scene_info.blocks
                    self.gaussians.points = scene_info.points
                else:
                    assert ("error dataset type!")
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, 
                                                              args.images, args.eval, 
                                                              args.img_skip_step)

        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")): 
            print("Found cameras_sphere.npz file, assuming blender mvs data set!")
            if args.gs_type == "gs_block":
                scene_info = sceneLoadTypeCallbacks["sphere_Block"](args.source_path, "cameras_sphere.npz",
                                                                 args.eval, opt.num_splats,
                                                                 opt.ratio_block_scene, opt.scale_block_min
                                                                 )
                self.gaussians.blocks = scene_info.blocks
                self.gaussians.points = scene_info.points
            else:
                scene_info = sceneLoadTypeCallbacks["sphere"](args.source_path, "cameras_sphere.npz",
                                                           args.white_background)  

        elif os.path.exists(os.path.join(args.source_path, "cameras.npz")):  # 1.
            print("Found cameras_sphere.npz file, assuming blender mvs data set!")
            if args.gs_type == "gs_block":
                scene_info = sceneLoadTypeCallbacks["sphere_Block"](args.source_path, "cameras.npz",  # _sphere
                                                                 args.eval, opt.num_splats,
                                                                 opt.ratio_block_scene, opt.scale_block_min
                                                                 )
                self.gaussians.blocks = scene_info.blocks
                self.gaussians.points = scene_info.points
            else:
                scene_info = sceneLoadTypeCallbacks["sphere"](args.source_path, "cameras.npz",
                                                           args.white_background)  




        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if args.gs_type == "gs_block":
                print("Found transforms_train.json file, assuming Blender_Block data set!")
                scene_info = sceneLoadTypeCallbacks["Blender_Block"](
                    args.source_path, args.white_background, args.eval, opt.num_splats,
                    opt.ratio_block_scene, opt.scale_block_min)
                self.gaussians.blocks = scene_info.blocks
                self.gaussians.points = scene_info.points
                self.points_gt = scene_info.points_gt
                self.scale_mat = scene_info.scale_mat
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)

        if not self.loaded_iter:
            if args.gs_type == "gs_block":
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  
            random.shuffle(scene_info.test_cameras)  

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                            resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras,
                                                                           resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,"point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            self.gaussians.point_cloud = scene_info.point_cloud
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


