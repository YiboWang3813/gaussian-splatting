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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        """ 
        Initialize a camera object.
        
        Args:
            resolution (tuple): The target resolution of image, e.g., (width, height) 
            colmap_id (int): camera id, 标识用哪台相机拍摄得到的 
            R (np.ndarray): Rotation matrix of this camera (3, 3)
            T (np.ndarray): Translation vector of this camera (3,)
            FoVx (float): Horizontal field of view of the camera in raduians
            FoVy (float): Vertical field of view of the camera in radians
            depth_params (dict): Parameters for depth estimation
            image (PIL.Image): The image captured by this camera 
            invdepthmap (np.ndarray): The inverse depth map of the image
            image_name (str): The name of the image
            uid (int): The unique identifier of this image 
            trans (np.ndarray): The translation vector to adjust the camera's center (3,)
            scale (float): The scale factor to adjust the camera's center
            data_device (str): The device on which the data is stored, e.g., 'cpu' or 'cuda'
            train_test_exp (bool): do training or testing 
            is_test_dataset (bool): 
            is_test_view (bool): 
        """
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

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]  # (3, H, W) 
        # Get alpha mask (1, H, W) 
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # Try to load inv depth map 
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0  # far plane 
        self.znear = 0.01  # near plane 

        self.trans = trans
        self.scale = scale 

        # Get the view transformation and projection transformation 
        # .transpose(0, 1) is because pytorch use row-major storage, while opengl use column-major storage
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()  # (4, 4)
        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).cuda()  # (4, 4)
        # Get the full projection transformation (view + projection)
        # in opengl manner, the order of matrix multiplication is reversed
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix 
        self.camera_center = self.world_view_transform.inverse()[3, :3]  # (3,)
        
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

