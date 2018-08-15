
/****************************************************************************************
 *
 * Created by the Numerical Algorithms Group Ltd, 2017.
 *
 ***************************************************************************************/

#pragma once

#include <vector>
#include "oskar_grid_wproj_fpga.hpp"

void oskar_process_all_tiles(
       const int num_w_planes,
       const int* support,
       const int oversample,
       const int* compact_wkernel_start,
       const float2* compact_wkernel,
       const double cell_size_rad,
       const double w_scale,
       const int grid_size,
       int boxTop_u, int boxTop_v,
       const int tileWidth,
       const int tileHeight,
       int numTiles_y, int numTiles_v,
       const int* numPointsInTiles,
       const int* offsetsPointsInTiles,
       const float* bucket_uu,
       const float* bucket_vv,
       const float* bucket_ww,
       const float2* bucket_vis,
       const int *workQueue_pu, const int *workQueue_pv,
       double* norm,
       float* grid
       );

void oskar_grid_wproj_cpu_f(
    const int num_w_planes,
    const int* support,
    const int oversample,
    const int conv_size_half,
    const float* conv_func,
    const int num_vis,
    const float* uu,
    const float* vv,
    const float* ww,
    const float* vis,
    const float* weight,
    const double cell_size_rad,
    const double w_scale,
    const int grid_size,
    size_t* num_skipped,
    double* norm,
    float* grid
    );

