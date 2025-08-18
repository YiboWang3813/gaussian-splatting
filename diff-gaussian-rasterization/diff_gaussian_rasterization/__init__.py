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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,  # torch.Tensor, background color, (3,)
            means3D,  # torch.Tensor, positions of gaussian primitives, (N, 3)
            colors_precomp,  # torch.Tensor, pre-computed colors of gaussian primitives, (N, 3)
            opacities, # torch.Tensor, opacities of gaussian primitives, (N, 1)
            scales,  # torch.Tensor, scaling factors of gaussian primitives, (N, 3)
            rotations,  # torch.Tensor, rotation quaternions of gaussian primitives, (N, 4)
            raster_settings.scale_modifier,  # float, 1.0 
            cov3Ds_precomp,  # torch.Tensor, pre-computed covariance matrices of gaussian primitives, (N, 6)
            raster_settings.viewmatrix,  # torch.Tensor, view transform matrix, (4, 4) 
            raster_settings.projmatrix,  # torch.Tensor, projection transformation matrix, (4, 4)
            raster_settings.tanfovx,  # float, tan of half Fovx
            raster_settings.tanfovy,  # float, tan of half Fovy 
            raster_settings.image_height,  # int, height of resized image 
            raster_settings.image_width,  # int, width of resized image 
            sh,  # torch.Tensor, shs of gaussian primitives, (N, 16, 3)
            raster_settings.sh_degree,  # int, active sh degree 
            raster_settings.campos,  # torch.Tensor, camera center, (3, )
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int  # resized image height, int 
    image_width: int  # resized image width, int 
    tanfovx : float  # float tan of half Fovx
    tanfovy : float  # float, tan of half Fovy 
    bg : torch.Tensor  # background color (3,) [0.0, 0.0, 0.0]
    scale_modifier : float  # 1.0 
    viewmatrix : torch.Tensor  # world-to-camera transform matrix (4, 3) 
    projmatrix : torch.Tensor  # projection matrix (4, 4)
    sh_degree : int  # current active sh degree, int 
    campos : torch.Tensor  # camera center, (3,)
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        """
        Forward the gaussian rasterizer and get the rendered image. 

        Args: 
            means3D (torch.Tensor): the position of 3d gaussian primitives, (N, 3) 
            means2D (torch.Tensor): gradients of the 2D (screen-space) means, (N, 2) 
            opacities (torch.Tensor): the opacity of the 3D gaussian primitives, (N, 1)
            shs (torch.Tensor): the SH coefficients of the 3D gaussian primitives, (N, 16, 3)
            colors_precomp (torch.Tensor): the precomputed RGB color of the 3D gaussian primitives, (N, 3)
            scales (torch.Tensor): the scaling factor of the 3D gaussian primitives, (N, 3)
            rotations (torch.Tensor): the rotation quaternion of the 3D gaussian primitives, (N, 4)
            cov3D_precomp (torch.Tensor): the precomputed 3D covariance matrix of the 3D gaussian primitives, (N, 6)
        
        Returns:
            rendered_image, radii, depth_image 
        """
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

