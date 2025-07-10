import math
import os
import numpy as np
import open3d as o3d
import torch
import trimesh
from einops import rearrange
from pytorch3d.structures import Meshes
from torch import nn

from games.block_mesh_splatting.utils.graphics_utils import BlockMeshPointCloud
from games.block_mesh_splatting.utils.pytorch import safe_pow
from games.block_mesh_splatting.utils.superquadric import parametric_sq, implicit_sq, \
    cal_cluster_label, sdf_cube, quaternion_to_rotation_matrix, fit_superquadric
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid, rot_to_quat_batch
from utils.loss_utils import l1_loss, ssim
from utils.plot import get_fancy_color
from utils.sh_utils import RGB2SH

OVERLAP_N_POINTS = 2048
OVERLAP_N_BLOCKS = 1.95  # 1.55
OVERLAP_N_BLOCKS_2 = 1.55
OVERLAP_TEMPERATURE = 0.005  # 0.005
OVERLAP_TEMPERATURE_OCC = 0.05
from torch.nn import functional as F


class BlockGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int, ratio_block_scene: float,
                 scale_block_min: float):

        super().__init__(sh_degree)
        self.ratio_block_scene = ratio_block_scene
        self.scale_block_min = scale_block_min
        self.point_cloud = None
        self._alpha = torch.empty(0)
        self._scale = torch.empty(0)
        self.alpha = torch.empty(0)  
        self.faces = torch.empty(0)
        self.vertices = torch.empty(0)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.coarse_learning = True
        self.triangles = None
        self._blocks_SRT = None

    def capture_block(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._alpha,
            self._scale,
            self.sq_r,
            self.sq_s,
            self.sq_t,
            self.sq_occ,
            self.sq_eps,
            self.faces,
            self.sq_eta,
            self.sq_omega,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore_block(self, model_args, training_args):
        (self.active_sh_degree,
         self._features_dc,
         self._features_rest,
         self._alpha,
         self._scale,
         self.sq_r,
         self.sq_s,
         self.sq_t,
         self.sq_occ,
         self.sq_eps,
         self.faces,
         self.sq_eta,
         self.sq_omega,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.n_blocks = self.sq_r.shape[0]
        self.update_alpha()
        self.prepare_scaling_rot()

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_opacity(self):
        return (self.opacity_activation(self.sq_occ).unsqueeze(dim=1)).\
                expand(-1, self.per_gs_num, -1).reshape(-1, 1)

    @property
    def get_mesh_scaling(self):
        return self.scaling_activation(self.sq_s) + self.scale_block_min

    @property
    def get_mesh_rot(self):
        return self.rotation_activation(self.sq_r)

    @property
    def get_mesh_trans(self):
        return self.sq_t

    def create_from_pcd(self, pcd: BlockMeshPointCloud, spatial_lr_scale: float):
        self.point_cloud = pcd
        self.n_blocks = len(pcd)
        self.spatial_lr_scale = spatial_lr_scale
        self._alpha = []
        self._features_dc = []
        self._features_rest = []
        self._scale = []
        self.sq_r = []
        self.sq_s = []
        self.sq_occ = []
        self.sq_t = []
        self.per_gs_num =  pcd[0].points.shape[0]
        max_radii2D = 0
        vertices_batch = []
        for p in pcd:
            alpha_point_cloud = p.alpha.float().cuda()
            self.gs_scale = 1/ math.sqrt(alpha_point_cloud.shape[1])
            scale = (self.gs_scale) * torch.ones((self.per_gs_num, 1)).float().cuda()
            vertices_batch.append(torch.tensor(p.vertices, device="cuda").float())

            fused_color = RGB2SH(torch.tensor(np.asarray(p.colors)).float().cuda())
            features = torch.zeros((self.per_gs_num, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
            occupancy = inverse_sigmoid(0.7 * torch.ones_like(p.occupancy, dtype=torch.float, device="cuda"))
            self._alpha.append(nn.Parameter(alpha_point_cloud))  # check update_alpha
            self._features_dc.append(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest.append(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scale.append(nn.Parameter(scale.requires_grad_(True)))
            self.sq_r.append(p.R_4d.requires_grad_(True))
            self.sq_s.append(p.S.requires_grad_(True))
            self.sq_t.append(p.T.requires_grad_(True))
            self.sq_occ.append(occupancy.requires_grad_(True))
            max_radii2D += self.per_gs_num
        print("num of sq:", len(self.sq_r), "num of gaussian:", len(self._alpha))
        self._alpha = nn.Parameter(torch.stack(self._alpha)).requires_grad_(False)
        self._scale = nn.Parameter(torch.stack(self._scale)).requires_grad_(False)
        self._features_dc = nn.Parameter(torch.cat(self._features_dc))
        self._features_rest = nn.Parameter(torch.cat(self._features_rest))
        self.sq_r = nn.Parameter(torch.cat(self.sq_r))
        self.sq_s = nn.Parameter(torch.cat(self.sq_s))
        self.sq_t = nn.Parameter(torch.cat(self.sq_t))
        self.sq_occ = nn.Parameter(torch.cat(self.sq_occ))
        self.sq_eps = nn.Parameter(torch.zeros((self.n_blocks, 2), device="cuda"))
        self.faces = self.blocks.faces_padded().to('cuda')
        self.vertices_batch = torch.stack(vertices_batch, dim=0)
        vertices_unit = self.vertices_batch / self.ratio_block_scene
        self.sq_eta = torch.asin(vertices_unit[..., 1])
        self.sq_omega = torch.atan2(vertices_unit[..., 0], vertices_unit[..., 2])
        self.max_radii2D = torch.zeros((max_radii2D), device="cuda")
        self.update_alpha()
        self.prepare_scaling_rot()


    def update_alpha(self):
        """

        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0
        """
        alpha = torch.relu(self._alpha) + 1e-8
        alpha = alpha.flatten(start_dim=0, end_dim=1)
        self.alpha = alpha / alpha.sum(dim=-1, keepdim=True)


    def get_verts(self):
        eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
        verts = parametric_sq(self.sq_eta, self.sq_omega, eps1, eps2) * self.ratio_block_scene
        self._blocks_eps = eps1, eps2
        return verts

    def get_meshes(self):
        return Meshes(self.vertices, self.faces)

    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid to 2nd vertex onto subspace spanned by v0 and v1
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        B, M, _ = self.faces.shape
        rotation_matrixs = quaternion_to_rotation_matrix(self.get_mesh_rot)
        self._blocks_SRT = (self.get_mesh_scaling, rotation_matrixs, self.get_mesh_trans)
        vertices = self.get_verts()

        self.vertices = torch.bmm(vertices * self._blocks_SRT[0].unsqueeze(1), self._blocks_SRT[1]) + \
                        self._blocks_SRT[2].unsqueeze(1)
        triangles = self.vertices[torch.arange(B)[:, None, None], self.faces]
        _xyz = torch.matmul(
            self.alpha,
            triangles.reshape(-1, *triangles.shape[2:])
        )
        _xyz = rearrange(_xyz, '(b m) c h -> b m c h', b=B)
        normals = torch.linalg.cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
            dim=2
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=2)
        v1 = triangles[:, :, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, :, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)
        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        scales = torch.concat((s1, s2), dim=2).unsqueeze(dim=2)
        scales = scales.broadcast_to((B, scales.shape[1], self.alpha.shape[-2], 2))
        _scaling = torch.log(torch.nn.functional.relu(self._scale * scales.flatten(start_dim=1, end_dim=2)) + eps)
        rotation = torch.stack((v1, v2, v0), dim=2).unsqueeze(dim=2)
        rotation = rotation.broadcast_to((B, scales.shape[1], self.alpha.shape[-2], 3, 3)).flatten(start_dim=1, end_dim=2)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)
        self._scaling = _scaling.reshape(-1, 2)
        self._xyz = rearrange(_xyz, 'b m c h -> b (m c) h').reshape(-1, 3)

    def filter_primitive(self, thr=0.2):
        B = self.sq_occ.shape[0]
        mask = self.opacity_activation(self.sq_occ) > thr
        mask_index = torch.nonzero(mask, as_tuple=True)[0]

        if len(mask_index) != B:
            print("primitive opacity:", self.opacity_activation(self.sq_occ))
            print('after filter_primitive:', len(mask_index))
            # learnable parameters
            mask_flattened = mask.expand(-1, self._features_dc.shape[0] // B).flatten(start_dim=0, end_dim=1)
            optimizable_tensors = self.get_tensors_to_optimizer(mask.squeeze(-1), mask_flattened)
            self.sq_r = optimizable_tensors["r_mesh"]
            self.sq_s = optimizable_tensors["s_mesh"]
            self.sq_t = optimizable_tensors["t_mesh"]
            self.sq_occ = optimizable_tensors["occupancy_mesh"]
            self.sq_eps = optimizable_tensors["sq_eps"]
            self._alpha = optimizable_tensors["alpha"]
            self._scale = optimizable_tensors["scaling"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]

            # non-leanable parameters
            self.sq_eta = self.sq_eta[mask_index]
            self.sq_omega = self.sq_omega[mask_index]
            self.vertices_batch = self.vertices_batch[mask_index]
            self.faces = self.faces[mask_index]
            self.n_blocks = len(mask_index)
            rotation_matrixs = quaternion_to_rotation_matrix(self.get_mesh_rot)
            self._blocks_SRT = (self.get_mesh_scaling, rotation_matrixs, self.get_mesh_trans)
            eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
            self._blocks_eps = eps1, eps2

        self.update_alpha()

    def cal_new_primitive(self, data_type):
        with torch.no_grad():
            S, R, T = self._blocks_SRT
            points = self.points[None].expand(self.n_blocks, -1, -1)
            points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
            eps1, eps2 = self._blocks_eps
            sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
            min_values = torch.min(sdf, dim=0).values  
            out_points = points[0, min_values > 5e-2, :]  # N,3
            if data_type == 'dtu':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(out_points.cpu().numpy())
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
                out_points = np.asarray(pcd.points)
                out_points = torch.from_numpy(out_points).float().cuda()
                indices = np.random.choice(out_points.shape[0], size=min(1_000, out_points.shape[0]), replace=False)
                out_points = out_points[indices]
                clusters, labels = cal_cluster_label(out_points, eps=0.15, min_samples=25)
            elif data_type == 'blendermvs':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(out_points.cpu().numpy())
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
                out_points = np.asarray(pcd.points)
                out_points = torch.from_numpy(out_points).float().cuda()
                indices = np.random.choice(out_points.shape[0], size=min(1_000, out_points.shape[0]), replace=False)
                out_points = out_points[indices]
                clusters, labels = cal_cluster_label(out_points, eps=0.15, min_samples=25)
            else:
                indices = np.random.choice(out_points.shape[0], size=min(1_000, out_points.shape[0]), replace=False)
                out_points = out_points[indices]
                clusters, labels = cal_cluster_label(out_points, eps=0.15, min_samples=25)

        superquadrics_S, superquadrics_R, superquadrics_T = [], [], []
        for cluster in clusters:
            if cluster != -1:
                cluster_points = out_points[labels == cluster]
                S, R, T = fit_superquadric(cluster_points, self.ratio_block_scene)
                superquadrics_S.append(S)
                superquadrics_R.append(R)
                superquadrics_T.append(T)
            torch.cuda.empty_cache()
        if len(superquadrics_S) > 0:
            return torch.cat(superquadrics_S), torch.cat(superquadrics_R), torch.cat(superquadrics_T)
        else:
            return None, None, None

    def add_primitive(self, data_type='dtu'):
        if self.n_blocks > 25:
            return None
        S, R, T = self.cal_new_primitive(data_type)
        if S is not None:
            n_add = S.shape[0]
            B = self.n_blocks
            new_features_dc = RGB2SH(0.5005 *
                                     torch.ones_like(rearrange(self._features_dc, '(b m) c h -> b m c h', b=B)[0:1]))
            new_features_dc = new_features_dc.repeat(n_add, 1, 1, 1)
            new_features_dc = new_features_dc.flatten(start_dim=0, end_dim=1)

            new_features_rest = torch.zeros_like(rearrange(self._features_rest, '(b m) c h -> b m c h', b=B)[0:1])
            new_features_rest = new_features_rest.repeat(n_add, 1, 1, 1)
            new_features_rest = new_features_rest.flatten(start_dim=0, end_dim=1)
            d = {"r_mesh": R,
                 "s_mesh": S,
                 "t_mesh": T,
                 "occupancy_mesh": inverse_sigmoid(0.7 * torch.ones((n_add, 1), dtype=torch.float, device="cuda")),
                 "sq_eps": torch.zeros((n_add, 2), device="cuda"),
                 "alpha": self._alpha[0:1].repeat(n_add, 1, 1, 1),
                 "f_dc": new_features_dc,
                 "f_rest": new_features_rest,
                 "scaling": self._scale[0:1].repeat(n_add, 1, 1)
                 }
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self.sq_r = optimizable_tensors["r_mesh"]
            self.sq_s = optimizable_tensors["s_mesh"]
            self.sq_t = optimizable_tensors["t_mesh"]
            self.sq_occ = optimizable_tensors["occupancy_mesh"]
            self.sq_eps = optimizable_tensors["sq_eps"]
            self._alpha = optimizable_tensors["alpha"]
            self._scale = optimizable_tensors["scaling"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]

            # non-leanable parameters
            self.sq_eta = torch.cat([self.sq_eta, self.sq_eta[0:1].repeat(n_add, 1)], dim=0)
            self.sq_omega = torch.cat([self.sq_omega, self.sq_omega[0:1].repeat(n_add, 1)], dim=0)

            self.vertices_batch = torch.cat([self.vertices_batch, self.vertices_batch[0:1].repeat(n_add, 1, 1)], dim=0)
            self.faces = torch.cat([self.faces, self.faces[0:1].repeat(n_add, 1, 1)], dim=0)
            self.n_blocks = self.sq_r.shape[0]
            rotation_matrixs = quaternion_to_rotation_matrix(self.get_mesh_rot)
            self._blocks_SRT = (self.get_mesh_scaling, rotation_matrixs, self.get_mesh_trans)
            eps1, eps2 = (self.sq_eps.sigmoid() * 1.8 + 0.1).split([1, 1], dim=-1)
            self._blocks_eps = eps1, eps2
            print('The num primitive is: ', self.n_blocks, 'add number is: ', n_add)
        
        self.update_alpha()

    def reset_params(self):
        occupancy_new = inverse_sigmoid(0.7 * torch.ones_like(self.sq_occ, dtype=torch.float, device="cuda"))
        optimizable_tensors = self.replace_tensor_to_optimizer(occupancy_new, "occupancy_mesh")
        self.sq_occ = optimizable_tensors["occupancy_mesh"]

        _scale_new = 0.5 * torch.ones_like(self._scale, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(_scale_new, "scaling")
        self._scale = optimizable_tensors["scaling"]

        _features_dc_new = RGB2SH(0.5005 * torch.ones_like(self._features_dc, dtype=torch.float, device="cuda"))
        optimizable_tensors = self.replace_tensor_to_optimizer(_features_dc_new, "f_dc")
        self._features_dc = optimizable_tensors["f_dc"]

        _features_rest_new = torch.zeros_like(self._features_rest, dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(_features_rest_new, "f_rest")
        self._features_rest = optimizable_tensors["f_rest"]

    def get_tensors_to_optimizer(self, mask_batch, mask_flattened):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] == 'f_dc' or group["name"] == 'f_rest':
                mask = mask_flattened
            else:
                mask = mask_batch
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

  
    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0] * self.get_xyz.shape[1], 1), device="cuda")
        l = [
            {'params': self.sq_r, 'lr': 5e-3, "name": "r_mesh"},
            {'params': self.sq_s, 'lr': 5e-3, "name": "s_mesh"},
            {'params': self.sq_t, 'lr': 5e-3, "name": "t_mesh"},
            {'params': self.sq_eps, 'lr': 1e-3, "name": "sq_eps"},
            {'params': self.sq_occ, 'lr': 1e-3, "name": "occupancy_mesh"},  
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._alpha], 'lr': 0., "name": "alpha"},
            {'params': [self._scale], 'lr': 0., "name": "scaling"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


    def activate_lr_gs(self, training_args):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == "alpha":
                param_group['lr'] = training_args.scaling_lr
            elif param_group['name'] == "scaling":
                param_group['lr'] = training_args.scaling_lr
            elif param_group['name'] == "f_dc":
                param_group['lr'] = training_args.feature_lr
            elif param_group['name'] == "f_rest":
                param_group['lr'] = training_args.feature_lr / 20.0
            else:
                param_group['lr'] = 0.


    def update_learning_rate(self, iteration, training_args) -> None:
        """ Learning rate scheduling per step """
        self.coarse_learning = iteration < training_args.coarse_learning_until_iter
        # self.use_primitive_coverage = iteration < training_args.primitive_coverage_until_iter
        # self.optimize_vertices = iteration >= training_args.optimize_vertices_from_iter

    def save_ply(self, path):
        print("primitive opacity:", self.opacity_activation(self.sq_occ))
        self._save_ply(path)
        attrs = self.__dict__
        additional_attrs = [
            '_alpha',
            '_scale',
            'vertices',
            'faces'
        ]
        save_dict = {}
        for attr_name in additional_attrs:
            save_dict[attr_name] = []
            for m in attrs[attr_name]:
                save_dict[attr_name].append(m)
        num_meshes = self.n_blocks
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        torch.save(save_dict, path_model)
        # color points set
        np.random.seed(42)
        colors = (get_fancy_color(num_meshes + 1).cpu().numpy() * 255).astype(np.uint8)
        vertices_list = save_dict['vertices']
        path_model = path.replace('point_cloud.ply', 'mesh.obj')
        meshes = []
        for i, faces in enumerate(self.faces):
            if self.opacity_activation(self.sq_occ[i]) > 0.1:
                mesh = trimesh.Trimesh(vertices=vertices_list[i].detach().cpu().numpy(),
                                       faces=faces.detach().cpu().numpy())
                mesh.visual.face_colors = [np.concatenate([colors[i], np.array([255])])] * len(mesh.faces)
                meshes.append(mesh)

        merged_mesh = trimesh.util.concatenate(meshes)
        merged_mesh.export(path_model)
        # self.draw_gaussians_disk(os.path.dirname(path_model), 0)

    def load_ply(self, path):
        self._load_ply(path)
        path_model = path.replace('point_cloud.ply', 'model_params.pt')
        params = torch.load(path_model)
        alpha = params['_alpha']
        scale = params['_scale']
        vertices = params['vertices']
        faces = params['faces']
        self._alpha = [nn.Parameter(a) for a in alpha]
        self._scale = [nn.Parameter(s) for s in scale]
        self.vertices = [nn.Parameter(v) for v in vertices]
        self.faces = faces


    def guassian_scale_regularization(self, control_scale=0.5):
        scale_reg_loss = (self._scale - control_scale).clamp(min=0.)
        return {'scale_reg': scale_reg_loss.mean()}

    def primitive_scale_regularization(self, control_scale=3.0):
        scale_reg_loss = (self.get_mesh_scaling - control_scale).clamp(min=0.)
        return {'scale_reg': scale_reg_loss.sum()}

    def overlap_loss(self):
        N = self.n_blocks
        with torch.no_grad():
            points = torch.rand(N, OVERLAP_N_POINTS, 3, device='cuda') * 2 - 1
            S, R, T = self._blocks_SRT
            points = (points * self.ratio_block_scene * S[:, None]) @ R + T[:, None]
            points = points.view(-1, 3)[None].expand(N, -1, -1)

        points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
        eps1, eps2 = self._blocks_eps
        sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)

        occupancy = torch.sigmoid(-sdf / OVERLAP_TEMPERATURE)
        if self.coarse_learning:
            alpha = self.opacity_activation(self.sq_occ + torch.randn_like(self.sq_occ))  #
            occupancy = occupancy * alpha
            overlap_loss = (occupancy.sum(0) - OVERLAP_N_BLOCKS).clamp(0).mean()

        else:
            alpha = self.opacity_activation(self.sq_occ + torch.randn_like(self.sq_occ))  # 
            occupancy = occupancy * alpha
            overlap_loss = (occupancy.sum(0) - OVERLAP_N_BLOCKS).clamp(0).mean()

        return {'overlap': overlap_loss}

    def parsimony_loss(self):
        loss_parsimony = safe_pow(self.opacity_activation(self.sq_occ), 0.5)
        return {'parsimony': loss_parsimony.mean()}



    def ray_uncoverage_loss(self, sdf, label_mask):
        sdf_rays = rearrange(sdf, 'b n c -> n b c').flatten(start_dim=1, end_dim=2)
        # out mask
        max_values = torch.max(-sdf_rays, dim=1).values
        max_values = max_values * (label_mask < 0.5)
        ray_out_mask = torch.clamp(max_values - 1e-3, min=0)
        num = (label_mask < 0.5).sum()
        result = {
            'ray_out_mask': ray_out_mask.sum() / num,
        }
        return result

    def occupancy_opacity_loss(self, sdf, label_mask):
        occupancy_alpha = self.opacity_activation(self.sq_occ)[..., None]
        occupancy_alpha = occupancy_alpha.expand(-1, sdf.shape[-2], sdf.shape[-1])
        occupancy_alpha = torch.where(sdf <= 1e-8, occupancy_alpha, 0)
        occupancy_alpha = rearrange(occupancy_alpha, 'b n c -> n b c').flatten(start_dim=1, end_dim=2)
        occupancy_alpha = torch.max(occupancy_alpha, dim=1).values
        occupancy_lossses_alpha = F.binary_cross_entropy(occupancy_alpha.clip(1e-3, 1.0 - 1e-3), label_mask)
        return {'primitive_opacity': occupancy_lossses_alpha.mean()}

    def ray_coverage_loss(self, sdf, label_mask):
        # sdf: b n c
        sdf_rays = rearrange(sdf, 'b n c -> n b c').flatten(start_dim=1, end_dim=2)
        min_values = torch.min(sdf_rays, dim=1).values
        ray_in_mask = torch.clamp(min_values[label_mask > 0.5] - 1e-3, min=0)
        result = {
            'ray_in_mask': ray_in_mask.mean(),
        }
        return result

    def summarize_losses(self, image, gt_image, viewpoint_cam, opt, use_reg=True):
        loss_dict = {}
        Ll1 = l1_loss(image, gt_image) * opt.rgb_weight
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if viewpoint_cam.original_mask is not None:
            mask = viewpoint_cam.original_mask.cuda()


        if opt.primtive_scale_weight and (opt.data_type == 'dtu' or opt.data_type == 'blendermvs'):
            losses_primtive_scale_dict = self.primitive_scale_regularization()
            loss_dict.update(losses_primtive_scale_dict)
            for loss_name, loss_item in losses_primtive_scale_dict.items():
                loss = loss + loss_item * opt.primtive_scale_weight

        if use_reg and opt.parsimony_weight > 0:  #
            losses_parsimony_dict = self.parsimony_loss()
            loss_dict.update(losses_parsimony_dict)
            for loss_name, loss_item in losses_parsimony_dict.items():
                loss = loss + loss_item * opt.parsimony_weight

        if use_reg and viewpoint_cam.original_mask is not None:
            if opt.data_type == 'dtu':
                near, far = 0.5, 3.5
            elif opt.data_type == 'blender':
                near, far = 2., 5.
            elif opt.data_type == 'blendermvs':
                near, far = 1, 3
            else:
                assert ("error dataset type")
            sampled_points, ray_idx = viewpoint_cam.cast_rays(depth_min=near, depth_max=far, num_samples=128)
            label_mask = mask.flatten(start_dim=0, end_dim=1)[ray_idx]
            sampled_points = sampled_points.flatten(start_dim=0, end_dim=1)
            points = sampled_points.expand(self.n_blocks, -1, -1)
            S, R, T = self._blocks_SRT
            points_inv = ((points - T[:, None]) @ R.transpose(1, 2)) / (S[:, None] * self.ratio_block_scene)
            eps1, eps2 = self._blocks_eps
            sdf = implicit_sq(points_inv, eps1, eps2, as_sdf=2)
            sampled_points_sdf = rearrange(sdf, 'b (n c)-> b n c', n=label_mask.shape[0])
            if opt.ray_coverage_weigth > 0:
                losses_ray_coverage_dict = self.ray_coverage_loss(sampled_points_sdf, label_mask)
                loss_dict.update(losses_ray_coverage_dict)
                for loss_name, loss_item in losses_ray_coverage_dict.items():
                    loss = loss + loss_item * opt.ray_coverage_weigth

            if opt.ray_uncoverage_weigth > 0:
                losses_ray_uncoverage_dict = self.ray_uncoverage_loss(sampled_points_sdf, label_mask)
                loss_dict.update(losses_ray_uncoverage_dict)
                for loss_name, loss_item in losses_ray_uncoverage_dict.items():
                    loss = loss + loss_item * opt.ray_uncoverage_weigth

            if opt.occupancy_weight > 0:  #
                losses_ray_coverage_dict = self.occupancy_opacity_loss(sampled_points_sdf, label_mask)
                loss_dict.update(losses_ray_coverage_dict)
                for loss_name, loss_item in losses_ray_coverage_dict.items():
                    loss = loss + loss_item * opt.occupancy_weight
            #

        if opt.overlap_weight > 0:  #
            losses_overlap_dict = self.overlap_loss()
            loss_dict.update(losses_overlap_dict)
            for loss_name, loss_item in losses_overlap_dict.items():
                loss = loss + loss_item * opt.overlap_weight

        return Ll1, loss, loss_dict

