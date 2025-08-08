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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array  # position 
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """ 
    Compute the transformation matrix from world coordinates to camera view coordinates.

    Args:
        R (np.array): Rotation matrix (3x3), 
        拍摄图像时,相机坐标系朝向相对于世界坐标系朝向的旋转矩阵
        t (np.array): Translation vector (3,), 
        拍摄图像时,相机坐标系原点在世界坐标系中的位置 
        translate (np.array): Additional translation to apply to the camera center (3,).
        scale (float): Scaling factor for the camera center.

    Returns:
        Rt (np.array): 4x4 transformation matrix from world to camera view coordinates.
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # 世界坐标系朝向到世界坐标系朝向的旋转矩阵 调整坐标系朝向
    Rt[:3, 3] = t  # 这应该是 -R.T @ t 
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt) 
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale # 调整相机中心 
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """ 获得从相机坐标系到NDC坐标系的投影矩阵 
    Args:
        znear (float): 近平面距离
        zfar (float): 远平面距离
        fovX (float): 水平方向的视场角（弧度）
        fovY (float): 垂直方向的视场角（弧度）
    Returns:
        P (torch.Tensor): 投影矩阵 (4, 4) """ 
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))