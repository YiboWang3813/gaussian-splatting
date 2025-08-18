/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;  // 右边得到的是N*16个元素的vec3数组 左边找首个元素位置再偏移
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 限制t的投影坐标不超过相机的视场角范围 由此更新t的坐标 
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;  // 相机水平和竖直方向上的约束范围 
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;  // 点t在水平和竖直方向上的投影
	t.x = min(limx, max(-limx, txtz)) * t.z;  // 约束点t在水平方向的投影在[-limx, limx]之间 
	t.y = min(limy, max(-limy, tytz)) * t.z;  // 约束点t在竖直方向的投影在[-limy, limy]之间

	// Build the jacobian matrix J  
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// Extract the matrix W from the view transform matrix 
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	// Recover the 3D covariance matrix using symmetry  
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// Compute the 2D covariance matrix 
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;  // Ensure the diagonal element of cov2D is not too small 
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };  // Discard 3rd row and column
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization. 计算三维协方差矩阵 
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.

/**
 * @brief Preprocesses data on the GPU using CUDA for further computations.
 *
 * This kernel processes 3D points and associated data to perform various operations
 * like scaling, rotating, and projecting the points, while also computing properties like
 * opacities, covariances, and colors in preparation for rendering or further processing.
 * The function runs in parallel across multiple CUDA threads, where each thread handles
 * a portion of the data.
 *
 * @tparam C The number of channels or components (depends on the specific use case, e.g., 1 for grayscale, 3 for RGB).
 *
 * @param P The number of points (size of the point cloud).
 * @param D The active sh degree.
 * @param M The max sh degree coefficients.
 * @param orig_points Means of all gaussian primitives in the world coordinate system. 
 * @param scales Scaling factors of all gaussian primitives.  
 * @param scale_modifier A modifier for the scaling.
 * @param rotations Rotation quaternions of all gaussian primitives. 
 * @param opacities Opacities of all gaussian primitives.
 * @param shs Spherical harmonics coefficients of all gaussian primitives.
 * @param clamped Clipping status of all Gaussian primitives' colors, arranged in RGB order. True means the color is negative.
 * @param cov3D_precomp Pre-computed 3D covariance matrices for all Gaussian primitives.
 * @param colors_precomp Pre-computed color values (RGB) for all Gaussian primitives.
 * @param viewmatrix View matrix to achieve world-to-camera transformation. 
 * @param projmatrix Full matrix to achieve world-to-clip space transformation. 
 * @param cam_pos Camera center in the world coordinate system.
 * @param W Image width. 
 * @param H Image height. 
 * @param tan_fovx The tangent of the horizontal field of view angle.
 * @param tan_fovy The tangent of the vertical field of view angle.
 * @param focal_x The focal length in the x-direction (usually in pixels).
 * @param focal_y The focal length in the y-direction (usually in pixels).
 * @param radii Radius of projected 2D circle in the pixel coodinate system of all gaussian primitives. 
 * @param points_xy_image Means of projected gaussian primitives in the pixel coordinate.
 * @param depths Depths of gaussian primitives in the camera system. 
 * @param cov3Ds Pointer to an array to store the 3D covariance matrices.
 * @param rgb Pointer to an array storing RGB values for each point.
 * @param conic_opacity Pointer to an array storing opacity values and inverse 2D covariance matrices.
 * @param grid The grid dimensions for the CUDA kernel.
 * @param tiles_touched Indicates whether the projection of a Gaussian ellipsoid touches a CUDA block.
 * @param prefiltered A flag indicating whether Gaussian ellipsoids that do not meet criteria should be pre-filtered (currently unused).
 */
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();  // 获得当前线程在grid中的全局索引 
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0; // 半径 
	tiles_touched[idx] = 0; // 是否接触图块 

	// Perform near culling, quit if outside. 判断当前点在不在视锥以内
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };  // 拿到idx点的世界坐标  
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);  // 将世界坐标变换到裁剪空间 
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w }; // 齐次除法w 转换到NDC坐标系

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 计算三维协方差矩阵 
	const float* cov3D;
	if (cov3D_precomp != nullptr) // 之前算过直接提取 
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else // 没算过就计算 再放到cov3Ds中 
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix 计算二维协方差矩阵 
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm) 计算cov2D的逆 
	float det = (cov.x * cov.z - cov.y * cov.y); // 计算cov的行列式 
	if (det == 0.0f) // 行列式为0 不进行后续计算 
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv }; // 按公式组合 

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 这里的tile就是block

	// 用cov2D计算得到这个2D高斯分布最大能覆盖的像素范围 用圆形来覆盖
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));  // 计算协方差矩阵的特征值 对应高斯分布椭圆的半轴长度
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));  // 根据特征值得到一个最大的覆盖范围
	
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) }; // 把均值从NDC坐标系转到像素坐标系 
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid); // 获得2d椭球的外接矩形在整个grid中所处的位置 block索引形式
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// 计算当前椭球的颜色 用球谐函数
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;  // 该高斯椭球在相机坐标系中的z坐标 
	radii[idx] = my_radius;  // 该高斯椭球投影到像素坐标系中得到的2D圆的半径 
	points_xy_image[idx] = point_image;  // 该高斯椭球在像素坐标系中的中心坐标
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] }; // 逆二维协方差矩阵 + 不透明度 
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); // 该高斯椭球是否触及图块 触及>0 不触及=0
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
/*
这个函数的计算逻辑是这样，一个thread对应一个pixel，这个thread会存在于一个block里，block和tile是一个概念，
一个block会对应很多个落在它身上的高斯椭球，需要通过循环把这些高斯椭球全遍历一次，对每次遍历到的高斯椭球都要取它的中心坐标，
二维协方差矩阵的逆矩阵以及不透明度来计算power，alpha以及衰减T并分别判断能否把它的颜色按一定比例混合到当前pixel的颜色上。
代码中为了减轻遍历的计算压力，提高计算效率，采用了分round的方式，但归根结底就是要把当前pixel所在block对应的全部高斯椭球全遍历一次
*/
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges, // 每个tile-i在point_list_key中的开始和结束索引 [start, end) 
	const uint32_t* __restrict__ point_list, // 每个tile对应哪个gaussian-idx
	int W, int H, // image width, height 
	const float2* __restrict__ points_xy_image, // 所有二维高斯椭球的中心坐标 有P个元素 
	const float* __restrict__ features, // 所有高斯椭球的颜色 有3*P个元素 
	const float4* __restrict__ conic_opacity, // 所有高斯椭球的Cov2D的逆矩阵和不透明度 有P个元素 
	float* __restrict__ final_T, // 每个pixel上最终的T值 有W*H个元素 
	uint32_t* __restrict__ n_contrib, // 每个pixel上有贡献的高斯椭球数量 有W*H个元素 
	const float* __restrict__ bg_color, // 背景颜色 有3个元素 分别表示对RGB三通道的背景色
	float* __restrict__ out_color) // 输出颜色 一维数组 按行优先排列的 有3*W*H个元素 
{
	// Identify current tile and associated min/max pixel range. 
	// According to current block and thread index, find the current pixel tackled by the current thread.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // block的左上角x,y坐标
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) }; // block的右下角x,y坐标 用W,H限制越界
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y }; // 当前thread处理的像素x,y坐标 
	uint32_t pix_id = W * pix.y + pix.x; // 当前pixel的ID 
	float2 pixf = { (float)pix.x, (float)pix.y }; 

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x]; // find the current block id, extract range
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE); // 每轮加载BLOCK_SIZE个Guassian到共享内存
	int toDo = range.y - range.x; // the number of gaussians need to do 

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE]; // 存id 
	__shared__ float2 collected_xy[BLOCK_SIZE]; // 存xy坐标 
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // 存cov2D的逆矩阵和不透明度 

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 }; // 存当前pixel的RGB颜色 

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank(); // 结合当前batch和当前thread id得到该thread在全局range中的位置
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress]; // gaussian idx 
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; // center position of current gaussian-j 
			float2 d = { xy.x - pixf.x, xy.y - pixf.y }; // 计算当前椭球中心和当前pixel的距离 
			float4 con_o = collected_conic_opacity[j]; // 当前椭球的逆二维协方差矩阵和不透明度 
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

			if (power > 0.0f) // power条件不满足 不能参与颜色叠加 
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 遵循指数衰减 从二维高斯椭球中心的不透明度opacity 结合exp(power) 计算得到pixel位置的alpha 
			float alpha = min(0.99f, con_o.w * exp(power)); // 用opacity和power计算alpha 
			
			if (alpha < 1.0f / 255.0f) // alpha条件不满足 不能参与颜色叠加 
				continue;
			
			float test_T = T * (1 - alpha); // 用1-alpha来衰减T 

			if (test_T < 0.0001f) // test_T条件不满足 不能参与颜色叠加
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T; // 先根据id找到j这个椭球的颜色 然后和alpha和T一起计算颜色 

			T = test_T; // 更新T 

			// Keep track of last range entry to update this pixel. 记录当前pixel上已经叠加了多少个椭球了
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T; // 记录当前pixel在抛雪球后的T值 
		n_contrib[pix_id] = last_contributor; // 记录当前pixel上一共被抛了多少个雪球 
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch]; // 添加背景颜色 用一个pixel/thread对应3个RGB的思路
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block, // cuda settings, grid and block 
	const uint2* ranges, // 每个tile-i在point_list_key中的开始和结束索引 [start, end)
	const uint32_t* point_list, // 每个tile对应哪个gaussian-idx
	int W, int H, // image width, height 
	const float2* means2D, // in pixel system, center position of gaussian-idx
	const float* colors, // color of gaussian-idx 
	const float4* conic_opacity, // inverse of cov2D + opacity of gaussian-idx 
	float* final_T, // accumulation alpha for each pixel 
	uint32_t* n_contrib, // number of contribution for each pixel 
	const float* bg_color, // background color 
	float* out_color) // output color 
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}