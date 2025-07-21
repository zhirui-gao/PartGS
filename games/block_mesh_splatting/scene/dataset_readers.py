import math
import os
import warnings
from typing import NamedTuple, Optional

import numpy as np
import open3d as o3d
import torch
from numpy import ndarray
from plyfile import PlyData
from pytorch3d.io import load_ply
from pytorch3d.structures.meshes import join_meshes_as_batch
import trimesh
from games.block_mesh_splatting.utils import poses_align
from games.block_mesh_splatting.utils.graphics_utils import BlockMeshPointCloud
from games.block_mesh_splatting.utils.mesh import get_icosphere, get_cube
from games.block_mesh_splatting.utils.superquadric import cal_cluster_label, fit_superquadric
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, \
    read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.dataset_readers import readColmapCameras
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    storePly,
    readDTUCameras
)


class SceneInfo(NamedTuple):
    point_cloud: NamedTuple
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    blocks: list
    points: ndarray
    points_gt: Optional[np.array] = None
    scale_mat: Optional[np.array] = None

from utils.sh_utils import SH2RGB

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices





def readNeRFSyntheticBlockMeshInfo(
        path, white_background, eval, num_splats=[100]*8, ratio_block_scene=0.25, scale_min=0.2, use_superquadric=True,
        extension=".png",
):
    print("Reading Training Transforms")
    mesh_path = os.path.join(path, 'model.obj')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    xyz_min = mesh.get_min_bound().min()
    xyz_max = mesh.get_max_bound().max()
    bound = max(abs(xyz_max), abs(xyz_min)) + 0.05
    scale_to_unit_sphere = 1 / (bound)
    mesh = mesh.scale(scale=scale_to_unit_sphere, center=np.array([0., 0., 0.]))
    pose_flip = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])  # [x,y,z]->[x,-z,y]
    mesh = mesh.rotate(R=pose_flip, center=np.array([0., 0., 0.]))
    pcd = mesh.sample_points_uniformly(100_000)
    pc_gt = np.asarray(pcd.points).astype(np.float32)
    scale_mat = np.eye(4).astype(np.float32)

    cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    hold = 2
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % hold != 0]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % hold == 0]

    print("Reading Mesh object")

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcds = []
    ply_paths = []

    if os.path.exists(os.path.join(path, "point_cloud.ply")):
        ply_data = PlyData.read(os.path.join(path, "point_cloud.ply"))
        vertices = ply_data['vertex']
        points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
        indices = np.random.choice(points.shape[0], size=min(100_000, points.shape[0]), replace=False)
        points = points[indices]
    else:
        num_pts = 1000_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        points = np.random.random((num_pts, 3)) * 2. - 1.
        # filter points by mask
        for train_info in train_cam_infos:
            points_new = points @ train_info.R + train_info.T
            K = np.zeros((3, 3))
            K[0, 0] = (train_info.width / 2.0) / math.tan(train_info.FovX / 2.0)
            K[1, 1] = (train_info.height / 2.0) / math.tan(train_info.FovY / 2.0)
            K[0, 2] = train_info.width / 2.0
            K[1, 2] = train_info.height / 2.0
            K[2, 2] = 1.0
            image_points = K @ points_new.T
            image_points /= image_points[2, :]
            x_coords = np.round(image_points[0, :]).astype(int)
            y_coords = np.round(image_points[1, :]).astype(int)
            mask = np.array(train_info.image)[..., -1]
            height, width = mask.shape
            out_of_view_indices = (x_coords < 0) | (x_coords >= width) | (y_coords < 0) | (y_coords >= height)
            out_of_view_indices_bigger = (x_coords < -0.2 * width) | (x_coords >= 1.2 * width) | (
                    y_coords < -0.2 * height) | (
                                                 y_coords >= 1.2 * height)
            points_out_of_view = points[(~out_of_view_indices_bigger) & out_of_view_indices]
            x_coords = x_coords[~out_of_view_indices]
            y_coords = y_coords[~out_of_view_indices]
            mask_values = mask[y_coords, x_coords]
            points = points[~out_of_view_indices][mask_values > 0.5]
            points = np.concatenate((points, points_out_of_view), axis=0)
            print(points.shape)

    total_pts = 0
    if use_superquadric:
        block = get_icosphere(level=2)
    else:
        block = get_cube(level=3)

    blocks = join_meshes_as_batch([block for _ in range(len(num_splats))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().cuda()
    clusters, labels = cal_cluster_label(points, num_k=len(num_splats))
    range_T = torch.max(points, dim=0)[0] - torch.min(points, dim=0)[0]
    for i, num in enumerate(num_splats):
        cluster_points = points[labels == i]
        vertices, faces = blocks.get_mesh_verts_faces(i)
        sq_eta = torch.asin(vertices[..., 1])
        sq_omega = torch.atan2(vertices[..., 0], vertices[..., 2])
        S_init, R_4d, translation = fit_superquadric(cluster_points, ratio_block_scene)
        translation = translation + (torch.randn_like(translation) / 4).clamp(-0.5, 0.5) * range_T
        occupancy = torch.ones((1, 1), device="cuda")
        vertices = vertices * ratio_block_scene
        triangles = vertices[faces]
        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts
        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = BlockMeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            R_4d=R_4d,
            T=translation.cuda(),
            S=S_init.cuda(),
            occupancy=occupancy
        )
        pcds.append(pcd)
    torch.cuda.empty_cache()
    print(
        f"Generating random point cloud ({total_pts})..."
    )
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths,
                           blocks=blocks,
                           points=points,
                           points_gt=pc_gt,
                           scale_mat=scale_mat)
    return scene_info


def readNeuSDTUBlockInfo(
        path, render_camera, eval, num_splats, ratio_block_scene, scale_min,
        use_superquadric=True, use_colmap_points=False
):
    print("Reading Training data")
    cam_infos, pc_gt, scale_mat = readDTUCameras(path, render_camera)
    #
    test_indexes = [35,37,39,40]#[25,26,30,36,37,38,39,41,42,57,58,63,64,66,73,88] #[0,1,2,4,5,9,22,28,34,38,42,51,57,58,60,68,76,79,80,88,89]#[43,44,45,46,47,48,49,50,51,52,53,54,55]  #[11,12,13,14,17,18,30,32, 44, 45]
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_indexes]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx  in test_indexes]

    print("Reading Mesh object")

    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcds = []
    ply_paths = []

    if use_colmap_points:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        ply_path = os.path.join(path, "sparse/0/points3D_agum.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
        centroid = np.mean(points, axis=0)
        R_colmap, T_colmap = [], []
        mask_folder = os.path.join(path, 'mask')
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                               images_folder=os.path.join(path, "images"), mask_folder=mask_folder)
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        train_cam_infos_colmap = [c for idx, c in enumerate(cam_infos) if idx not in test_indexes]
        for cam_info in train_cam_infos_colmap:
            R_colmap.append(torch.from_numpy(np.transpose(cam_info.R)).float())
            T_colmap.append(torch.from_numpy(cam_info.T).float())
        R_colmap = torch.stack(R_colmap)
        T_colmap = torch.stack(T_colmap)
        poses_colmap = torch.cat([R_colmap, T_colmap[...,None]], dim=-1)

        R, T = [], []
        for cam_info in train_cam_infos:
            R.append(torch.from_numpy(np.transpose(cam_info.R)).float())
            T.append(torch.from_numpy(cam_info.T).float())
        R = torch.stack(R)
        T = torch.stack(T)
        poses = torch.cat([R, T[..., None]], dim=-1)
        sim3= poses_align.prealign_cameras(poses_colmap, poses, poses_colmap.device)
        # move colmap points to gt pose cordinates
        points = poses_align.transform_points(torch.from_numpy(points), sim3)
        # save points
        storePly(os.path.join(path, "point_cloud.ply"), points, np.ones((points.shape[0], 3)) * 255)
        points = points.cuda()
    else:
        if os.path.exists(os.path.join(path, "point_cloud.ply")):
            ply_data = PlyData.read(os.path.join(path, "point_cloud.ply"))
            vertices = ply_data['vertex']
            points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
            indices = np.random.choice(points.shape[0], size=min(100_000, points.shape[0]), replace=False)
            points = points[indices]
        else:
            num_pts = 200_000
            print(f"Generating random point cloud ({num_pts})...")
            # We create random points inside the bounds of the synthetic Blender scenes
            points = np.random.random((num_pts, 3)) * 1.4-0.7 #2.6-1.3 mvs
            filter_counts = np.zeros(num_pts)
            original_indices = np.arange(num_pts)

            # filter points by mask
            for train_info in train_cam_infos:
                points_new = points @ train_info.R + train_info.T
                K = np.zeros((3, 3))
                K[0, 0] = (train_info.width / 2.0) / math.tan(train_info.FovX / 2.0)
                K[1, 1] = (train_info.height / 2.0) / math.tan(train_info.FovY / 2.0)
                K[0, 2] = train_info.width / 2.0
                K[1, 2] = train_info.height / 2.0
                K[2, 2] = 1.0
                image_points = K @ points_new.T
                image_points /= image_points[2, :]
                x_coords = np.round(image_points[0, :]).astype(int)
                y_coords = np.round(image_points[1, :]).astype(int)
                mask = np.array(train_info.image)[..., -1]
                import cv2
                mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
                height, width = mask.shape
                out_of_view_indices = (x_coords < 0) | (x_coords >= width) | (y_coords < 0) | (y_coords >= height)
                out_of_view_indices_bigger = (x_coords < -0.5 * width) | (x_coords >= 1.5 * width) | (
                            y_coords < -0.5 * height) | (
                                                     y_coords >= 1.5 * height)
                # filter_counts[original_indices[out_of_view_indices_bigger]] += 1
                #
                # index = np.where(filter_counts[original_indices[out_of_view_indices_bigger]])

                points_out_of_view = points[(~out_of_view_indices_bigger) & out_of_view_indices]
                x_coords = x_coords[~out_of_view_indices]
                y_coords = y_coords[~out_of_view_indices]
                mask_values = mask[y_coords, x_coords]
                points = points[~out_of_view_indices][mask_values > 1e-2 ]#
                points = np.concatenate((points, points_out_of_view), axis=0)
                print(points.shape)


    total_pts = 0
    if use_superquadric:
        block = get_icosphere(level=2)
    else:
        block = get_cube(level=3)

    blocks = join_meshes_as_batch([block for _ in range(len(num_splats))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().cuda()
    clusters, labels = cal_cluster_label(points, num_k=len(num_splats))
    range_T = torch.max(points, dim=0)[0] - torch.min(points, dim=0)[0]
    for i, num in enumerate(num_splats):
        cluster_points = points[labels == i]
        vertices, faces = blocks.get_mesh_verts_faces(i)
        sq_eta = torch.asin(vertices[..., 1])
        sq_omega = torch.atan2(vertices[..., 0], vertices[..., 2])
        S_init, R_4d, translation = fit_superquadric(cluster_points, ratio_block_scene)
        translation = translation + (torch.randn_like(translation) / 4).clamp(-0.5, 0.5) * range_T

        occupancy = torch.ones((1, 1), device="cuda")
        vertices = vertices * ratio_block_scene
        triangles = vertices[faces]
        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts
        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = BlockMeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            R_4d=R_4d,
            T=translation.cuda(),
            S=S_init.cuda(),
            occupancy=occupancy
        )
        pcds.append(pcd)
    torch.cuda.empty_cache()
    print(
        f"Generating random point cloud ({total_pts})..."
    )
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths,
                           blocks=blocks,
                           points=points,
                           points_gt=pc_gt,
                           scale_mat=scale_mat)
    return scene_info

def readColmapBlockMeshInfo(
        path, images, eval, num_splats=[100]*8, ratio_block_scene=0.25, scale_min=0.2,
        llffhold=2,render_camera=None, use_colmap_points=False):
    if render_camera is not None:
        camera_dict = np.load(os.path.join(path, render_camera))
        tag = os.path.basename(path)
        filename = 'stl{}_total.ply'.format(tag.replace('scan', '').zfill(3))
        if os.path.exists(os.path.dirname(path) + '/Points/stl/' + filename):
            points = load_ply(os.path.dirname(path) + '/Points/stl/' + filename)[0]
            scale_mat = camera_dict['scale_mat_0']
            scale_inv = np.linalg.inv(scale_mat)
            pc_gt = points @ scale_inv[:3, :3] + scale_inv[:3, 3]
        else:
            scale_mat, pc_gt = None, None
            warnings.warn(f"Ground truth ply file does not exist: {os.path.dirname(path)}/Points/stl/{filename}")

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if use_colmap_points:  # whether to use colmap points, that does not matter !
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
    else:
        num_pts = 1000_000
        print(f"Generating random point cloud ({num_pts})...")
        points = np.random.random((num_pts, 3)) * 2. - 1.
        for train_info in train_cam_infos:
            points_new = points @ train_info.R + train_info.T
            K = np.zeros((3, 3))
            K[0, 0] = (train_info.width / 2.0) / math.tan(train_info.FovX / 2.0)
            K[1, 1] = (train_info.height / 2.0) / math.tan(train_info.FovY / 2.0)
            K[0, 2] = train_info.width / 2.0
            K[1, 2] = train_info.height / 2.0
            K[2, 2] = 1.0
            image_points = K @ points_new.T
            image_points /= image_points[2, :]
            # 获取投影到图像上的像素坐标
            x_coords = np.round(image_points[0, :]).astype(int)
            y_coords = np.round(image_points[1, :]).astype(int)
            mask = np.array(train_info.image)[..., -1]
            # 检查像素坐标是否在掩码范围内
            height, width = mask.shape
            out_of_view_indices = (x_coords < 0) | (x_coords >= width) | (y_coords < 0) | (y_coords >= height)
            out_of_view_indices_bigger = (x_coords < -0.2 * width) | (x_coords >= 1.2 * width) | (y_coords < -0.2 * height) | (
                    y_coords >= 1.2 * height)
            points_out_of_view = points[(~out_of_view_indices_bigger) & out_of_view_indices]
            x_coords = x_coords[~out_of_view_indices]
            y_coords = y_coords[~out_of_view_indices]
            mask_values = mask[y_coords, x_coords]
            points = points[~out_of_view_indices][mask_values > 0.5]
            points = np.concatenate((points, points_out_of_view), axis=0)

       
        cc = trimesh.PointCloud(points)
        cc.export(os.path.join(path, "./input.ply"))

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcds = []
    ply_paths = []
    total_pts = 0

    block = get_icosphere(level=2)

    blocks = join_meshes_as_batch([block for _ in range(len(num_splats))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().cuda()
    clusters, labels = cal_cluster_label(points, num_k=len(num_splats))
    range_T = torch.max(points,dim=0)[0] - torch.min(points,dim=0)[0]
    for i, num in enumerate(num_splats):
        cluster_points = points[labels == i]
        vertices, faces = blocks.get_mesh_verts_faces(i)
        sq_eta = torch.asin(vertices[..., 1])
        sq_omega = torch.atan2(vertices[..., 0], vertices[..., 2])
        S_init, R_4d, translation = fit_superquadric(cluster_points, ratio_block_scene)
        translation = translation + (torch.randn_like(translation)/4).clamp(-0.5, 0.5) * range_T
        occupancy = torch.ones((1, 1), device="cuda")
        vertices = vertices * ratio_block_scene
        triangles = vertices[faces]
        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts
        # We create random points inside the bounds traingles
        u = torch.sqrt(torch.rand(triangles.shape[0], num_pts_each_triangle, 1))
        v = torch.rand(triangles.shape[0], num_pts_each_triangle, 1)
        alpha = torch.cat([1 - u, u * (1 - v), u * v], dim=-1)

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = BlockMeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            R_4d = R_4d,
            T = translation.cuda(),
            S = S_init.cuda(),
            occupancy= occupancy
        )
        pcds.append(pcd)
    torch.cuda.empty_cache()
    print(
        f"Generating random point cloud ({total_pts})..."
    )
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths,
                           blocks=blocks,
                           points=points,
                           points_gt=pc_gt,
                           scale_mat=scale_mat)

    return scene_info



def readReplicaBlockMeshInfo(path, images, eval, num_splats=[100]*8, ratio_block_scene=0.25, scale_min=0.2, use_superquadric=True, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []



    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    points = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=-1)
    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcds = []
    ply_paths = []
    total_pts = 0
    if use_superquadric:
        block = get_icosphere(level=2)
    else:
        block = get_cube(level=3)
    blocks = join_meshes_as_batch([block for _ in range(len(num_splats))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    points = np.asarray(pcd.points)
    points = torch.from_numpy(points).float().cuda()
    clusters, labels = cal_cluster_label(points, num_k=len(num_splats))
    for i, num in enumerate(num_splats):
        cluster_points = points[labels == i]
        vertices, faces = blocks.get_mesh_verts_faces(i)
        sq_eta = torch.asin(vertices[..., 1])
        sq_omega = torch.atan2(vertices[..., 0], vertices[..., 2])
        S_init, R_4d, translation = fit_superquadric(cluster_points, ratio_block_scene)
        occupancy = torch.ones((1, 1), device="cuda")
        vertices = vertices * ratio_block_scene
        triangles = vertices[faces]
        num_pts_each_triangle = num
        num_pts = num_pts_each_triangle * triangles.shape[0]
        total_pts += num_pts
        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        )
        alpha = alpha/alpha.sum(dim=-1, keepdim=True)
        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = BlockMeshPointCloud(
            alpha=alpha,
            points=xyz,
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            vertices=vertices,
            faces=faces,
            transform_vertices_function=transform_vertices_function,
            triangles=triangles.cuda(),
            R_4d = R_4d,
            T = translation.cuda(),
            S = S_init.cuda(),
            occupancy= occupancy
        )
        pcds.append(pcd)
    torch.cuda.empty_cache()
    print(
        f"Generating random point cloud ({total_pts})..."
    )
    scene_info = SceneInfo(point_cloud=pcds,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_paths,
                           blocks=blocks,
                           points=points)

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Block": readNeRFSyntheticBlockMeshInfo,
    "DTU_Block": readNeuSDTUBlockInfo,
    "Colmap_Block": readColmapBlockMeshInfo,
    "Replica_Block":readReplicaBlockMeshInfo
}

