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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            """ 
            从缩放和旋转中构建协方差矩阵 
            Parameters: 
                scaling (torch.Tensor): 缩放因子 (N, 3) 
                scaling_modifier 
                rotation (torch.Tensor): 旋转四元数 (N, 4) 
            Returns:
                symm (torch.Tensor): 保留独有元素的协方差矩阵 (N, 6) 
            """
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2) # 构建协方差矩阵cov (N, 3, 3)
            symm = strip_symmetric(actual_covariance) # 去掉cov中的重复值 (N, 6) 
            return symm
        
        # 定义激活函数 
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid # 不透明度 保证0-1之间 靠近0透明 靠近1不透明
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize # 按行求元素平方后的和开根号 再除以这个结果

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

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        """ 从点云文件中初始化全部高斯椭球的参数 
        Args: 
            pcd: 点云数据 一个实例化的点云对象 """
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() # (N, 3)
        """ 初始化球谐函数系数 """
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) # RGB转球谐函数的系数 仅含第0阶 (N, 3)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # 全部高斯椭球的球谐函数系数 (N, 3, 16)
        features[:, :3, 0 ] = fused_color # 初始化 把0阶放进去 
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        """ 初始化缩放因子和旋转因子 """
        # 计算距离 每个点都找它最近的3个邻居 把它和这3个邻居的距离的均值返回 应该返回的是平方 
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001) # (N,)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3) # 缩放因子 (N, 3) log相当于inverse_scale_activation
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # 旋转因子 (N, 4) 
        rots[:, 0] = 1 # 初始化为单位四元数 让w=1 cos1=0 

        """ 初始化不透明度 """
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")) # 不透明度 (N, 1)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True)) # 空间位置 (N, 3)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)) # 球谐函数系数的直流分量 (N, 1, 3)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)) # 球谐函数系数的高阶分量 (N, 15, 3)
        self._scaling = nn.Parameter(scales.requires_grad_(True)) # 缩放因子 (N, 3)
        self._rotation = nn.Parameter(rots.requires_grad_(True)) # 旋转因子 (N, 4)
        self._opacity = nn.Parameter(opacities.requires_grad_(True)) # 不透明度 (N, 1)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 二维投影后高斯分布的最大半径 
        
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.spatial_lr_scale = spatial_lr_scale

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 梯度累计值 (N, 1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 梯度累计值的分母 (N, 1)

        # 设置要优化的参数和优化器 
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ] # 针对不同参数设置不同的学习率 训练过程中会动态调整学习率

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 设置scheduler来控制xyz和exposure的学习率  
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step 按训练iteration更新学习率 '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        """ 返回一个全属性的字符串列表 """
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
        """ 把训练的结果存到.ply文件中 """
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

    def load_ply(self, path, use_train_test_exp = False):
        """ 加载点云文件(checkpoint)并赋值给训练参数 """
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
        """ 重置不透明度 原论文中提到每隔一段时间要重置一次不透明度 """
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ------------------------------ 改optimizer的操作 增加或减少高斯椭球 ------------------------------ #
    def replace_tensor_to_optimizer(self, tensor, name):
        """ 把优化器里的一个param替换 并同时清零它的exp_avg和exp_avg_sq """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor) # 动量
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) # 二次动量 

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """ 修改优化器的全部param 找到每个param把N轴按mask进行去留 """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask] # 修改state

                del self.optimizer.state[group['params'][0]] # 删除旧的state 
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True))) # 更新param tensor
                self.optimizer.state[group['params'][0]] = stored_state # 替换为新的state 

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        """ 把新的张量添加到optimizer中 实现增加高斯椭球 """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ------------------------------ 自适应密度控制部分 ------------------------------ #
    def prune_points(self, mask):
        """ 修改点 按照mask沿着N轴进行去留 实现删除高斯椭球 
        顺序是先改了optimizer 再从optimizer提取改了的tensor 最后用改了的tensor替换类的变量 """
        valid_points_mask = ~mask # valid_mask=1 mask=0 这个点保留 valid_mask=0 mask=1 这个点删掉
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

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        """ 添加新的高斯椭球到优化器 """
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation} # 新高斯椭球按属性装进字典 

        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 把新高斯椭球的参数加入optimizer 
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] # 更新现有高斯椭球的属性 

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii)) # radii在原来基础上增加 
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 累计梯度归零 
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 最大2D半径也归零 

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """ 密度化和分裂操作 把梯度过大并超过场景范围的高斯椭球删掉 替换为小的高斯椭球
        Args: 
            grads: 当前梯度 
            grad_threshold: 梯度阈值 
            scene_extent: 场景范围 
            N: 规定1个大的高斯椭球要分裂成N个小高斯椭球 """
        
        # 找到满足梯度条件的高斯椭球 即这个高斯椭球的梯度大于要求的阈值 
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda") # 存储全部高斯椭球现在的梯度 (N,)
        padded_grad[:grads.shape[0]] = grads.squeeze() # 把grads的数据添加到padded_grad中 
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False) # (N,)
        # 找到满足缩放条件的高斯椭球 即这个高斯椭球的最大缩放因子值大于比例因子*场景范围
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 计算要更新的高斯椭球的空间位置 先计算一个偏置再叠加到原来的空间位置上
        stds = self.get_scaling[selected_pts_mask].repeat(N,1) # 标准差 (N*N_mask, 3) N_mask是符合条件的高斯椭球数量
        means = torch.zeros((stds.size(0), 3), device="cuda") # 均值 (N*N_mask, 3)
        samples = torch.normal(mean=means, std=stds) # 采样点 (N*N_mask, 3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1) # 旋转 (N*N_mask, 3, 3)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1) # (N*N_mask, 3)  
        
        # 计算要更新的高斯椭球的其他属性 
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # filter中 原来梯度过大并超出场景范围的标记为1 其余为0 新创建的也为0 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 删除了这些梯度过大并超出场景范围的高斯椭球 保留了梯度和大小合适的高斯椭球以及新的小高斯椭球
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """ 克隆操作 复制一些符合条件的旧高斯并添加进去 
        Args: 
            grads: 当前梯度 
            grad_threshold: 梯度阈值 
            scene_extent: 场景范围 """
        # Extract points that satisfy the gradient condition 满足梯度大于阈值 
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 并且大小小于场景范围的 
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 找到这些旧的小高斯 复制一份添加到原有的高斯椭球集合中 
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        """ 密度化并修改 自适应密度控制的接口函数 
        Args: 
            max_grad: 梯度阈值 满足条件的高斯椭球的梯度要大于它 
            min_opacity: 不透明度最小值 筛选删掉不透明度太小的高斯椭球 
            extent: 场景范围 筛选删掉空间尺寸过大的高斯椭球 
            max_screen_size: 最大屏幕尺寸 筛选删除投影到屏幕上过大的高斯椭球 
            radii """
        grads = self.xyz_gradient_accum / self.denom # 计算平均梯度 (N, 1)
        grads[grads.isnan()] = 0.0 # 把梯度为无穷大的地方置为0 

        self.tmp_radii = radii

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent) # 进行clone和split 

        prune_mask = (self.get_opacity < min_opacity).squeeze() # 删除不透明度低于阈值的 (N,)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # 删除屏幕上尺寸过大的 (N,)
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 删除在空间上尺寸过大的 (N,)
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask) # 统一剔除不透明度、2D尺寸过大、空间尺寸过大的高斯椭球 

        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """ 添加密度化统计信息 修改update_filter对应高斯椭球的梯度和分母 """
        # 把viewspace在x和y方向上的梯度累加到xyz_gradient_accum中 
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # 每处理一次 计数器+1 
        self.denom[update_filter] += 1
