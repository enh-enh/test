#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// 假设utils.h包含get_tile_index、get_output_shape、get_tiling_info、get_U_shape、get_V_shape等函数声明
// 如果实际中这些函数定义在其他地方，需要确保其正确性和可访问性
#include "utils.h"

//// 定义新的FLT_H和FLT_W
//#define FLT_H 3
//#define FLT_W 3
//
//// 定义新的tile尺寸
//#define TILE_IN_H 6
//#define TILE_IN_W 6
//#define TILE_OUT_H 6
//#define TILE_OUT_W 6

// 新的image_transform函数，适配F(6*6,3*3)分片
void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*packed_image_tensor_t)[TILE_IN_W][collapsed_dim_size];
	typedef float(*V_tensor_t)[TILE_IN_W][collapsed_dim_size];
	packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
	V_tensor_t V_tensor = (V_tensor_t)V;
	float z0, z1, z2, z3, z4, z5;
	// 遍历collapsed_dim_size
	for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
		// 遍历tile的宽度
		for (int64_t w = 0; w < TILE_IN_W; ++w) {
			// 根据F(6*6,3*3)的计算规则进行变换
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			z4 = 0;
			z5 = 0;
			for (int64_t i = 0; i < 3; i++) {
				z0 += packed_image_tensor[i][w][idx] * (i == 0? 4 : (i == 1? -5 : 1));
				z1 += packed_image_tensor[i][w][idx] * (i == 0? -4 : (i == 1? -4 : 1));
				z2 += packed_image_tensor[i][w][idx] * (i == 0? 4 : (i == 1? -4 : 1));
				z3 += packed_image_tensor[i][w][idx] * (i == 0? -2 : (i == 1? -1 : 1));
				z4 += packed_image_tensor[i][w][idx] * (i == 0? 2 : (i == 1? -1 : 1));
				z5 += packed_image_tensor[i][w][idx] * (i == 0? 4 : (i == 1? -5 : 1));
			}
			V_tensor[0][w][idx] = z0;
			V_tensor[1][w][idx] = z1;
			V_tensor[2][w][idx] = z2;
			V_tensor[3][w][idx] = z3;
			V_tensor[4][w][idx] = z4;
			V_tensor[5][w][idx] = z5;
		}
		// 遍历tile的高度
		for (int64_t h = 0; h < TILE_IN_H; ++h) {
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			z4 = 0;
			z5 = 0;
			for (int64_t i = 0; i < 3; i++) {
				z0 += V_tensor[h][i][idx] * (i == 0? 4 : (i == 1? -5 : 1));
				z1 += V_tensor[h][i][idx] * (i == 0? -4 : (i == 1? -4 : 1));
				z2 += V_tensor[h][i][idx] * (i == 0? 4 : (i == 1? -4 : 1));
				z3 += V_tensor[h][i][idx] * (i == 0? -2 : (i == 1? -1 : 1));
				z4 += V_tensor[h][i][idx] * (i == 0? 2 : (i == 1? -1 : 1));
				z5 += V_tensor[h][i][idx] * (i == 0? 4 : (i == 1? -5 : 1));
			}
			V_tensor[h][0][idx] = z0;
			V_tensor[h][1][idx] = z1;
			V_tensor[h][2][idx] = z2;
			V_tensor[h][3][idx] = z3;
			V_tensor[h][4][idx] = z4;
			V_tensor[h][5][idx] = z5;
		}
	}
}

// 新的filter_transform函数，适配F(6*6,3*3)分片
void filter_transform(float *__restrict__ packed_filter,
                      float *__restrict__ U,
                      const filter_shape_t fs,
                      const U_shape_t us,
                      const int64_t collapsed_dim_size) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*packed_filter_tensor_t)[FLT_W][collapsed_dim_size];
	typedef float(*U_tensor_t)[TILE_IN_W][collapsed_dim_size];
	packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
	U_tensor_t U_tensor = (U_tensor_t)U;
	float z0, z1, z2, z3, z4, z5;
	// 遍历collapsed_dim_size
	for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
		// 遍历filter的宽度
		for (int64_t w = 0; w < FLT_W; ++w) {
			// 根据F(6*6,3*3)的计算规则进行变换
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			z4 = 0;
			z5 = 0;
			for (int64_t i = 0; i < 3; i++) {
				z0 += packed_filter_tensor[i][w][idx] * (i == 0? 1.0f / 4 : (i == 1? -1.0f / 6 : 0));
				z1 += packed_filter_tensor[i][w][idx] * (i == 0? -1.0f / 6 : (i == 1? -1.0f / 6 : 0));
				z2 += packed_filter_tensor[i][w][idx] * (i == 0? -1.0f / 6 : (i == 1? 1.0f / 6 : 0));
				z3 += packed_filter_tensor[i][w][idx] * (i == 0? 1.0f / 24 : (i == 1? 1.0f / 12 : 0));
				z4 += packed_filter_tensor[i][w][idx] * (i == 0? 1.0f / 24 : (i == 1? -1.0f / 12 : 0));
				z5 += packed_filter_tensor[i][w][idx] * (i == 0? 0 : (i == 1? 1 : 0));
			}
			U_tensor[0][w][idx] = z0;
			U_tensor[1][w][idx] = z1;
			U_tensor[2][w][idx] = z2;
			U_tensor[3][w][idx] = z3;
			U_tensor[4][w][idx] = z4;
			U_tensor[5][w][idx] = z5;
		}
		// 遍历U的高度
		for (int64_t h = 0; h < TILE_IN_H; ++h) {
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			z4 = 0;
			z5 = 0;
			for (int64_t i = 0; i < 3; i++) {
				z0 += U_tensor[h][i][idx] * (i == 0? 1.0f / 4 : (i == 1? -1.0f / 6 : 0));
				z1 += U_tensor[h][i][idx] * (i == 0? -1.0f / 6 : (i == 1? -1.0f / 6 : 0));
				z2 += U_tensor[h][i][idx] * (i == 0? -1.0f / 6 : (i == 1? 1.0f / 6 : 0));
				z3 += U_tensor[h][i][idx] * (i == 0? 1.0f / 24 : (i == 1? 1.0f / 12 : 0));
				z4 += U_tensor[h][i][idx] * (i == 0? 1.0f / 24 : (i == 1? -1.0f / 12 : 0));
				z5 += U_tensor[h][i][idx] * (i == 0? 0 : (i == 1? 1 : 0));
			}
			U_tensor[h][0][idx] = z0;
			U_tensor[h][1][idx] = z1;
			U_tensor[h][2][idx] = z2;
			U_tensor[h][3][idx] = z3;
			U_tensor[h][4][idx] = z4;
			U_tensor[h][5][idx] = z5;
		}
	}
}

// 新的output_transform函数，适配F(6*6,3*3)分片
void output_transform(float *__restrict__ M,
                      float *__restrict__ Y,
                      const tiling_info_t ti,
                      const int64_t collapsed_dim_size) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*M_tensor_t)[TILE_IN_W][collapsed_dim_size];
	typedef float(*Y_tensor_t)[TILE_IN_W][collapsed_dim_size];
	M_tensor_t M_tensor = (M_tensor_t)M;
	Y_tensor_t Y_tensor = (Y_tensor_t)Y;
	float z0, z1, z2, z3;
	// 遍历collapsed_dim_size
	for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
		// 遍历tile的宽度
		for (int64_t w = 0; w < TILE_IN_W; ++w) {
			// 根据F(6*6,3*3)的计算规则进行变换
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			for (int64_t i = 0; i < 6; i++) {
				z0 += M_tensor[i][w][idx];
				z1 += (i == 1? M_tensor[i][w][idx] : (i == 3? 2 * M_tensor[i][w][idx] : (i == 4? -2 * M_tensor[i][w][idx] : 0)));
				z2 += (i == 1? -M_tensor[i][w][idx] : (i == 2? M_tensor[i][w][idx] : (i == 3? 4 * M_tensor[i][w][idx] : (i == 4? 4 * M_tensor[i][w][idx] : 0))));
				z3 += (i == 1? -M_tensor[i][w][idx] : (i == 2? -M_tensor[i][w][idx] : (i == 3? 8 * M_tensor[i][w][idx] : (i == 4? -8 * M_tensor[i][w][idx] : (i == 5? M_tensor[i][w][idx] : 0)))));
			}
			Y_tensor[0][w][idx] = z0;
			Y_tensor[1][w][idx] = z1;
			Y_tensor[2][w][idx] = z2;
			Y_tensor[3][w][idx] = z3;
		}
		// 遍历tile的高度
		for (int64_t h = 0; h < TILE_OUT_H; ++h) {
			z0 = 0;
			z1 = 0;
			z2 = 0;
			z3 = 0;
			for (int64_t i = 0; i < 6; i++) {
				z0 += Y_tensor[h][i][idx];
				z1 += (i == 1? Y_tensor[h][i][idx] : (i == 3? 2 * Y_tensor[h][i][idx] : (i == 4? -2 * Y_tensor[h][i][idx] : 0)));
				z2 += (i == 1? -Y_tensor[h][i][idx] : (i == 2? Y_tensor[h][i][idx] : (i == 3? 4 * Y_tensor[h][i][idx] : (i == 4? 4 * Y_tensor[h][i][idx] : 0))));
				z3 += (i == 1? -Y_tensor[h][i][idx] : (i == 2? -Y_tensor[h][i][idx] : (i == 3? 8 * Y_tensor[h][i][idx] : (i == 4? -8 * Y_tensor[h][i][idx] : (i == 5? Y_tensor[h][i][idx] : 0)))));
			}
			Y_tensor[h][0][idx] = z0;
			Y_tensor[h][1][idx] = z1;
			Y_tensor[h][2][idx] = z2;
			Y_tensor[h][3][idx] = z3;
		}
	}
}

// 新的filter_packing函数，适配F(6*6,3*3)分片
void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*filter_tensor_t)[fs.ic][FLT_H][FLT_W];
	typedef float(*packed_filter_tensor_t)[FLT_W][fs.oc][fs.ic];
	filter_tensor_t filter_tensor = (filter_tensor_t)filter;
	packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
	#pragma omp parallel
	{
		#pragma omp for
		for (int64_t h = 0; h < FLT_H; ++h)
			for (int64_t w = 0; w < FLT_W; ++w)
				for (int64_t oc = 0; oc < fs.oc; oc++)
					for (int64_t ic = 0; ic < fs.ic; ic++)
						packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
	}
}

// 新的image_packing函数，适配F(6*6,3*3)分片
void image_packing(float *__restrict__ image,
                   float *__restrict__ packed_image,
                   const image_shape_t is,
                   const tiling_info_t ti) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*packedImage_tensor_t)[TILE_IN_W][ti.num_tiles][is.ic];
	typedef float(*image_tensor_t)[is.ic][is.h][is.w];
	packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
	image_tensor_t image_tensor = (image_tensor_t)image;
	#pragma omp parallel
	{
		#pragma omp for
		for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
			for (int64_t ic = 0; ic < is.ic; ic++) {
				for (int64_t h = 0; h < TILE_IN_H; ++h) {
					for (int64_t w = 0; w < TILE_IN_W; ++w) {
						tile_index_t tidx = get_tile_index(tile, ti);
						int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
						if (hh * 6 + h < is.h && ww * 6 + w < is.w)
							packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 6 + h)][(ww * 6 + w)];
						else
							packed_image_tensor[h][w][tile][ic] = 0;
					}
				}
			}
		}
	}
}

// 新的output_unpacking_store函数，适配F(6*6,3*3)分片
void output_unpacking_store(float *__restrict__ Y,
                            float *__restrict__ out,
                            const out_shape_t os,
                            const tiling_info_t ti) {
	// 根据新的分片尺寸调整数据类型定义
	typedef float(*Y_tensor_t)[TILE_IN_W][os.oc][ti.num_tiles];
	typedef float(*out_tensor_t)[os.oc][os.h][os.w];
	Y_tensor_t Y_tensor = (Y_tensor_t)Y;
	out_tensor_t out_tensor = (out_tensor_t)out;
	#pragma omp parallel
	{
		#pragma omp for
		for (int64_t h = 0; h < TILE_OUT_H; ++h) {
			for (int64_t w = 0; w < TILE_OUT_W; ++w) {
				for (int64_t oc = 0; oc < os.oc; oc++) {
					for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
						tile_index_t tidx = get_tile_index(tile, ti);
						int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
						if (hh * 6 + h < os.h && ww * 6 + w < os.w)
							out_tensor[batch][oc][(hh * 6 + h)][(ww * 6 + w)] = Y_tensor[h][w][oc][tile];
					}
				}
			}
		}
	}
}

// 矩阵乘法函数
void sgemm(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
	typedef float(*A_tensor_t)[K];
	typedef float(*B_tensor_t)[K];
	typedef float(*C_tensor_t)[M];
	A_tensor_t A_tensor = (A_tensor_t)A;
	B_tensor_t B_tensor = (B_tensor_t)B;
	C_tensor_t C_tensor = (C_tensor_t)C;
	for (int64_t m = 0; m < M; ++m) {
		for (int64_t n = 0; n < N; ++n) {
			C_tensor[n][m] = 0;
			for (int64_t k = 0; k < K; ++k) {
				C_tensor[n][m] += A_tensor[m][k] * B_tensor[n][k];
			}
		}
	}
}

// Winograd卷积函数
void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
	// 定义新的形状变量
	const image_shape_t is = {.bs = batch_num,.ic = input_channel_num,.h = image_height,.w = image_width};
	const filter_shape_t fs = {.oc = output_channel_num,.ic = input_channel_num,.h = FLT_H,.w = FLT_W};
	const out_shape_t os = get_output_shape(is, fs);
	const tiling_info_t ti = get_tiling_info(is, os);
	const U_shape_t us = get_U_shape(fs, ti);
	const V_shape_t vs = get_V_shape(is, ti);

	float *packed_filter = (float *)malloc(sizeof(float) * FLT_H * FLT_W * fs.oc * fs.ic);
	float *packed_image = (float *)malloc(sizeof(float) * TILE_IN_H * TILE_IN_W * ti.num_tiles * is.ic);
	float *U = (float *)malloc(sizeof(float) * TILE_IN_H * TILE_IN_W * us.oc * us.ic);
	float *V = (float *)malloc(sizeof(float) * TILE_IN_H * TILE_IN_W * vs.num_tiles * vs.ic);
	float *M = (float *)malloc(sizeof(float) * TILE_IN_H * TILE_IN_W * us.oc * vs.num_tiles);
	float *Y = (float *)malloc(sizeof(float) * TILE_OUT_H * TILE_IN_W * os.oc * ti.num_tiles);

	filter_packing(filter, packed_filter, fs);
	filter_transform(packed_filter, U, fs, us, us.oc * us.ic);
	image_packing(image, packed_image, is, ti);
	image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

	#pragma omp parallel
	{
		typedef float(*U_tensor_t)[TILE_IN_W][us.oc][us.ic];
		typedef float(*V_tensor_t)[TILE_IN_W][vs.num_tiles][vs.ic];
		typedef float(*M_tensor_t)[TILE_IN_W][us.oc][vs.num_tiles];
		#pragma omp for
		for (int64_t h = 0; h < TILE_IN_H; ++h) {
			for (int64_t w = 0; w < TILE_IN_W; ++w) {
				U_tensor_t U_tensor = (U_tensor_t)U;
				V_tensor_t V_tensor = (V_tensor_t)V;
				M_tensor_t M_tensor = (M_tensor_t)M;
				sgemm(vs.num_tiles,
				      us.oc,
				      us.ic,
				      (float *)(V_tensor[h][w]),
				      (float *)(U_tensor[h][w]),
				      (float *)(M_tensor[h][w]));
			}
		}
	}

	output_transform(M, Y, ti, us.oc * vs.num_tiles);
	output_unpacking_store(Y, out, os, ti);

	free(packed_filter);
	free(packed_image);
	free(U);
	free(V);
	free(M);
	free(Y);
}
