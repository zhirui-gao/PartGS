
import math
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.K = torch.zeros((3, 3), dtype=torch.float32).to(self.data_device)
        self.K[0, 0] = (self.image_width / 2.0) / math.tan(self.FoVx / 2.0)
        self.K[1, 1] = (self.image_height / 2.0) / math.tan(self.FoVy / 2.0)
        self.K[0, 2] = self.image_width / 2.0
        self.K[1, 2] = self.image_height / 2.0
        self.K[2, 2] = 1.0

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            # self.gt_alpha_mask = torch.from_numpy(gt_alpha_mask[None]).to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            self.original_mask = (gt_alpha_mask[0]).float().to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.original_mask= torch.ones((self.image_height, self.image_width),device=self.data_device)



        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv, t_inv[..., None]], dim=-1)
        return pose_inv

    def cam2world(self, X, world_view_transform):
        X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
        pose_inv = self.invert(world_view_transform.transpose(-1,-2)[:3,:])
        return X_hom @ pose_inv.transpose(-1,-2)


    def cast_rays(self, depth_min=0, depth_max=10, num_samples=100,num_rays=2048):
        # Generate pixel grid
        ray_idx = torch.randperm(self.image_height * self.image_width, device=self.data_device)[:num_rays]
        xy_grid = (torch.stack([ray_idx % self.image_width, ray_idx // self.image_width],
                               dim=-1).float().add_(0.5)) # [HW,2]
        xy_grid_hom = torch.cat([xy_grid, torch.ones_like(xy_grid[..., :1])], dim=-1)
        # Apply inverse projection to get points in camera space
        points = xy_grid_hom @ self.K.inverse().transpose(-1, -2)
        grid_3D = self.cam2world(points, self.world_view_transform)
        center_3D = torch.zeros_like(grid_3D)
        center_3D = self.cam2world(center_3D, self.world_view_transform)
        ray = grid_3D - center_3D
        rand_samples = torch.rand(ray.shape[0],num_samples, 1, device = self.data_device)
        rand_samples += torch.arange(num_samples, device=self.data_device)[None, :, None].float()

        depth_samples = rand_samples / num_samples * (depth_max - depth_min) + depth_min
        sampled_points = center_3D.unsqueeze(1) + ray.unsqueeze(1) * depth_samples
        return sampled_points, ray_idx

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

