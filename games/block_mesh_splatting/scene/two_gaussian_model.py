
import os

import numpy as np
import pytorch3d
import torch
from einops import rearrange
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from games.block_mesh_splatting.utils.superquadric import implicit_sq
from games.block_mesh_splatting.utils.superquadric import parametric_sq, quaternion_to_rotation_matrix
from scene.gaussian_model import GaussianModel
from utils.general_utils import get_expon_lr_func, build_rotation
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class TwoGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int, ratio_block_scene:float=0.25, scale_block_min:float=0.2):
        super().__init__(sh_degree)
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)
        self.scale_block_min = scale_block_min
        self.ratio_block_scene = ratio_block_scene

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_semantic(self):
        return self._semantic

    def create_from_block(self, model_args, filter_inner_points=True):
        (self.active_sh_degree,
         _features_dc,
         _features_rest,
         _alpha,
         _scale,
         _R_4d,
         _S,
         _T,
         _occupancy,
         sq_eps,
         faces,
         sq_eta,
         sq_omega,
         max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        
        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        self.part_num = _R_4d.shape[0]
        print("Number of points at initialisation : ", _features_dc.shape[0])

        rotation_matrixs = quaternion_to_rotation_matrix(self.rotation_activation(_R_4d))
        self._blocks_SRT = (self.scaling_activation(_S).detach() + self.scale_block_min, rotation_matrixs.detach(), _T.detach() )

        eps1, eps2 = (sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        self._blocks_eps = eps1.detach(), eps2.detach()
        vertices = parametric_sq(sq_eta, sq_omega, eps1, eps2) * self.ratio_block_scene

        vertices = torch.bmm(vertices * self._blocks_SRT[0].unsqueeze(1), self._blocks_SRT[1]) + \
                        self._blocks_SRT[2].unsqueeze(1)
        triangles = vertices[torch.arange(self.part_num)[:, None, None], faces]

        alpha = torch.relu(_alpha) + self.eps_s0
        alpha = alpha.flatten(start_dim=0, end_dim=1)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        # sample the second dim of alpha
        alpha = alpha[:,::5,:]
        _scale = _scale[:,::5,:]
        _xyz = torch.matmul(
            alpha,
            triangles.reshape(-1, *triangles.shape[2:])
        )
        _xyz = rearrange(_xyz, '(b m) c h -> b (m c) h', b=self.part_num)
        normals = torch.linalg.cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
            dim=2
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + self.eps_s0)
        means = torch.mean(triangles, dim=2)
        v1 = triangles[:, :, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + self.eps_s0
        v1 = v1 / v1_norm
        v2_init = triangles[:, :, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + self.eps_s0)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        # s0 = self.eps_s0 * torch.ones_like(s1)
        scales = torch.concat((s1, s2), dim=2).unsqueeze(dim=2)
        scales = scales.broadcast_to((self.part_num, scales.shape[1], alpha.shape[-2], 2))
        _scaling = torch.log(
            torch.nn.functional.relu(_scale * scales.flatten(start_dim=1, end_dim=2)) + self.eps_s0)  # B, N,3

        rotation = torch.stack((v1, v2, v0), dim=2).unsqueeze(dim=2)
        rotation = rotation.broadcast_to((self.part_num, scales.shape[1], alpha.shape[-2], 3, 3))\
            .flatten(start_dim=1, end_dim=2)
        rotation = rotation.transpose(-2, -1)
        rotation = rot_to_quat_batch(rotation)

        opacities = (_occupancy[:, None, :]).repeat(1, _xyz.shape[1], 1).reshape(-1,1)
        semantics = torch.zeros((_xyz.shape[0], _xyz.shape[1], self.part_num), dtype=torch.float, device="cuda")
        for i in range(self.part_num):
            semantics[i, :, i] = 1.
        points_mask = self.filter_inter_points(_xyz).flatten(0, 1) > 0
        if filter_inner_points is False:
            points_mask = torch.ones_like(points_mask)

        # calculate the distance from initial points to the superquadric

        points = self._xyz.data  # initla points (such as colmap points)
        S, R, T = self._blocks_SRT
        points = points[None].expand(self.part_num, -1, -1)
        points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
        eps1, eps2 = self._blocks_eps
        sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
        # use the distance to determine the part id
        part_id = torch.argmin(sdf.transpose(0,1), dim=-1)
        _semantic = torch.zeros((self._xyz.shape[0], self.part_num), dtype=torch.float, device="cuda")
        _semantic[torch.arange(self._xyz.shape[0]), part_id] = 1.

        _xyz = torch.cat((self._xyz.data, _xyz.reshape(-1, 3)[points_mask]), dim=0)
        _features_dc = torch.cat((self._features_dc.data, _features_dc[::5,...][points_mask]), dim=0)
        _features_rest = torch.cat((self._features_rest.data, _features_rest[::5,...][points_mask]), dim=0)
        _scaling = torch.cat((self._scaling.data, _scaling.reshape(-1, 2)[points_mask]), dim=0)
        _rotation = torch.cat((self._rotation.data, rotation[points_mask]), dim=0)
        _opacity = torch.cat((self._opacity.data, opacities[points_mask]), dim=0)    
        _semantic = torch.cat((_semantic, semantics.reshape(-1, self.part_num)[points_mask]), dim=0)

        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self._semantic = _semantic
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
 

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._semantic = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") + 0.

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._semantic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         self._semantic,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._semantic.shape[1]):
            l.append('semantic_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def _save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        semantics = self._semantic.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, semantics, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def _load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        semantics_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("semantic_")]
        semantics_names = sorted(semantics_names, key=lambda x: int(x.split('_')[-1]))
        semantics = np.zeros((xyz.shape[0], len(semantics_names)))
        for idx, attr_name in enumerate(semantics_names):
            semantics[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic = torch.tensor(semantics,dtype=torch.float, device="cuda")
        # self._semantic = nn.Parameter(torch.tensor(semantics,dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._semantic = self._semantic[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_semantics, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = torch.cat((self._semantic, new_semantics), dim=0)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:, :1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_semantic, new_scaling,
                                   new_rotation)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_semantic,
                                   new_scaling, new_rotation)


    def inter_part_loss(self):
        loss = 0.
        indices = np.random.choice(self._xyz.shape[0], size=min(1_000, self._xyz.shape[0]), replace=False)
        points = self._xyz[indices]
        semantic = self._semantic[indices]
        points = points[None].expand(self.part_num, -1, -1)
        S, R, T = self._blocks_SRT
        points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
        eps1, eps2 = self._blocks_eps
        sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
        for i in range(self.part_num):
            loss = loss + (-sdf[i][semantic.argmax(dim=-1)!=i]).clamp(min=0.).mean()
        return loss

    @torch.no_grad()
    def filter_inter_points(self, xyz):
        S, R, T = self._blocks_SRT
        eps1, eps2 = self._blocks_eps
        mask_out = []
        for i in range(0, xyz.shape[0]):
            points = xyz[i][None].expand(self.part_num, -1, -1)
            points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
            sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
            points_out = sdf > -1e-2
            mask = torch.ones_like(points_out[0])
            for j in range(0, xyz.shape[0]):
                if i==j:
                    continue
                mask = mask * points_out[j]
            mask_out.append(mask)
        return torch.stack(mask_out,dim=0)


    @torch.no_grad()
    def filter_points_by_part(self,xyz):
        mask_out = []
        for i in range(0, xyz.shape[0]):
            mask = torch.ones_like(xyz[i,:,0])
            if i in [0]: # [1,6]
                mask = torch.zeros_like(xyz[i,:,0])
            mask_out.append(mask)
        return torch.stack(mask_out, dim=0)



