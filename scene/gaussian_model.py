#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from typing import Dict, List, Tuple

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # type: (torch.Tensor, float, torch.Tensor) -> torch.Tensor
            """ 
            Build the covariance matrix from the scaling and rotation parameters.

            Args: 
                scaling: (N, 3), scaling factors 
                scaling_modifier: scaling modifier
                rotation: (N, 4), rotation quaternions 

            Returns:
                symm: (N, 6), unique elements of the coveriance matrix  
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # (N, 3, 3) 
            symm = strip_symmetric(actual_covariance) # (N, 6) 
            return symm
        
        # activation functions 
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize 

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0 # 当前球谐函数的阶数 
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree # 球谐函数的最高阶数  
        self._xyz = torch.empty(0) # 椭球位置 
        self._features_dc = torch.empty(0) # 球谐函数的直流分量 
        self._features_rest = torch.empty(0) # 球谐函数的高阶分量 
        self._scaling = torch.empty(0) # 缩放因子 
        self._rotation = torch.empty(0) # 旋转因子 
        self._opacity = torch.empty(0) # 不透明度 
        self.max_radii2D = torch.empty(0) # 投影到平面后二维高斯分布的最大半径 
        self.xyz_gradient_accum = torch.empty(0) # 点云位置 梯度累积值 
        self.denom = torch.empty(0) # 统计的分母数量 xyz_gradient_accum要除以denom得到平均梯度 
        self.optimizer = None
        self.percent_dense = 0 # 百分比密度 和密度控制有关
        self.spatial_lr_scale = 0 # 对空间位置学习率的缩放因子 
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
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
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, cam_infos: int, spatial_lr_scale: float):
        """ 
        Create many gaussian primitives from the point cloud data. 

        Args: 
            pcd: point cloud object 
            cam_infos: camera information 
            spatial_lr_scale: spatial learning rate scaling factor 
        """
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # (N, 3)
        
        # Initialize the spherical harmonics (SH) coefficients
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # (N, 3) 
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # (N, 3, 16)
        features[:, :3, 0 ] = fused_color # Assign the base color to the first SH coefficient (degree 0)
        features[:, 3:, 1:] = 0.0 # Set higher-order SH coefficients to 0 

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Initialize the scaling factors and rotation quaternions 
        # Compute the squared distance from each point to its three nearest neighbors
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # (N,)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3) # scaling factors (N, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # rotation quaternions (N, 4)
        rots[:, 0] = 1 # Set the real part of the quaternion to 1

        # Initialize the opacities
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # (N, 1)

        # Initialize and parameterize all Gaussian primitives
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))  # position (N, 3)
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))  # DC component of SH coefficients (N, 1, 3)
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))  # higher-order SH components (N, 15, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True))  # scaling factors (N, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True))  # rotation quaternions (N, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True))  # opacity values (N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # maximum 2D projected radius (N,)
        
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.spatial_lr_scale = spatial_lr_scale  # scaling factor for spatial learning rate 

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # position gradient accumulation (N, 1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # normalization term for position gradients (N, 1)

        # Set needed parameters and an optimizer 
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ] 

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # Set schedulers to control learing rate of position and exposure 
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """ Return a list of all attributes of one gaussian primitive. """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz'] # x,y,z是点的坐标 nx,ny,nz是法向量 
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]): # 1*3
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]): # 15*3
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]): # 3
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]): # 4
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        """ Save the state of all gaussian primitives to a ply file. """
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, use_train_test_exp=False):
        """ Load the ply file and parameterize the gaussian primitives. """
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

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
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def reset_opacity(self):
        """ Reset opacity. """
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ------------------------------ Update optimizer, increase or decrease the gaussians ------------------------------ #
    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> Dict[str, torch.Tensor]:
        """
        Replace an existing tensor in the optimizer with a new one.

        Args:
            tensor: New tensor to be optimized.
            name: Name of the parameter group to update.

        Returns:
            Dictionary containing the updated optimizable tensor.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)

                # Reinitialize optimizer state (momentum and squared gradients)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # Replace parameter
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def _prune_optimizer(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prune all optimizer parameters along the N axis based on the given mask.

        Args:
            mask: Boolean mask indicating which elements to keep. Shape: (N,)

        Returns:
            Dictionary containing pruned and optimizable tensors.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            
            if stored_state is not None:
                # Prune optimizer state (momentum and squared gradients)
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # Replace parameter with pruned version
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Concatenate new tensors to optimizer parameters, enabling the addition of new Gaussian primitives.

        Args:
            tensors_dict: A dictionary of tensors to append, keyed by parameter group name.

        Returns:
            Dictionary containing updated optimizable tensors.
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                # Extend optimizer state with zeros for the new parameters
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                # Replace parameter with concatenated version
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )

            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # ------------------------------ Adaptive Density Control ------------------------------ #
    def prune_points(self, mask: torch.Tensor) -> None:
        """
        Remove Gaussian primitives based on a boolean mask along the N axis.

        Args:
            mask: Boolean mask where True indicates the point to be removed. Shape: (N,)
        """
        valid_points_mask = ~mask  # Invert mask: True = keep, False = remove
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_tmp_radii: torch.Tensor,
    ) -> None:
        """
        Add new Gaussian primitives to the current set.
        """
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        # Append new parameters to the optimizer
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        # Update internal parameter references
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Concatenate new radii
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        # Reset gradient accumulation and max projected radii
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float, N=2):
        """
        Densify and split Gaussians whose gradients are large and sizes exceed the scene extent.

        Args:
            grads (torch.Tensor): Gradients for current Gaussians. Shape: (N,)
            grad_threshold (float): Gradient threshold for splitting.
            scene_extent (float): Global scene size to determine large Gaussians.
            N (int): Number of new Gaussians to create from each selected one.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # Select Gaussians with high gradient and large spatial scale
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask &= torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent

        # Sample new positions around selected Gaussians
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # (N_mask * N, 3)
        samples = torch.normal(mean=0.0, std=stds)  # (N_mask * N, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)  # (N_mask * N, 3, 3) 
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)  # (N_mask * N, 3)

        # Prepare properties for new Gaussians
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # Append new Gaussians
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # Remove original Gaussians that were split
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Clone selected Gaussians that meet gradient and scale conditions by duplicating and adding them.

        Args:
            grads (torch.Tensor): Current gradients, shape (N,).
            grad_threshold (float): Threshold to select Gaussians with large gradients.
            scene_extent (float): Scene size used to filter Gaussians by scale.
        """
        # Select points with gradient norm above threshold and scale within limit
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask &= torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent

        # Extract parameters of selected Gaussians to clone
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # Append cloned Gaussians to existing ones
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """
        Adaptive densification and pruning of Gaussians based on gradient, opacity, and size.

        Args:
            max_grad (float): Gradient threshold; Gaussians with gradient above this will be densified.
            min_opacity (float): Minimum opacity threshold; Gaussians below this will be pruned.
            extent (float): Scene extent; used to filter out Gaussians that are too large spatially.
            max_screen_size (float): Max allowed screen-space size; prune Gaussians exceeding this projection size.
            radii (torch.Tensor): Radii tensor used for internal tracking.
        """
        grads = self.xyz_gradient_accum / self.denom  # Average gradients (N, 1)
        grads[grads.isnan()] = 0.0  # Replace NaNs with zero

        self.tmp_radii = radii

        # Clone and split Gaussians based on gradients and extent
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Build pruning mask for low opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            # Prune Gaussians too large on screen or in world space
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, torch.logical_or(big_points_vs, big_points_ws))

        self.prune_points(prune_mask)  # Remove Gaussians based on combined mask

        self.tmp_radii = None
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        Update densification statistics by accumulating gradient norms and counts for selected Gaussians.

        Args:
            viewspace_point_tensor (torch.Tensor): Points in view space with gradients.
            update_filter (torch.BoolTensor): Mask selecting Gaussians to update.
        """
        # Accumulate gradient norms on x and y axes
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        # Increment update counts
        self.denom[update_filter] += 1
