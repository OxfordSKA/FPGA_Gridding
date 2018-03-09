/*
 * Copyright (c) 2016-2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "oskar_grid_wproj.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Defines.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_grid_wproj_d(
        const size_t num_w_planes,
        const int* restrict support,
        const int oversample,
        const int conv_size_half,
        const double* restrict conv_func,
        const size_t num_points,
        const double* restrict uu,
        const double* restrict vv,
        const double* restrict ww,
        const double* restrict vis,
        const double* restrict weight,
        const double cell_size_rad,
        const double w_scale,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        double* restrict grid)
{
    size_t i;
    const size_t kernel_dim = conv_size_half * conv_size_half;
    const int grid_centre = grid_size / 2;
    const double grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const double pos_u = -uu[i] * grid_scale;
        const double pos_v = vv[i] * grid_scale;
        const double ww_i = ww[i];
        const double conv_conj = (ww_i > 0.0) ? -1.0 : 1.0;
        const size_t grid_w = (size_t)round(sqrt(fabs(ww_i * w_scale)));
        const int grid_u = (int)round(pos_u) + grid_centre;
        const int grid_v = (int)round(pos_v) + grid_centre;

        /* Get visibility data. */
        //const double weight_i = weight[i];
        const double weight_i = 1.0;
        const double v_re = weight_i * vis[2 * i];
        const double v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)round((round(pos_u) - pos_u) * oversample);
        const int off_v = (int)round((round(pos_v) - pos_v) * oversample);

        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];
        const size_t kernel_start = grid_w < num_w_planes ?
                grid_w * kernel_dim : (num_w_planes - 1) * kernel_dim;

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            *num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j)
        {
            size_t p1, t1;
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            t1 = abs(off_v + j * oversample);
            t1 *= conv_size_half;
            t1 += kernel_start;
            for (k = -w_support; k <= w_support; ++k)
            {
                size_t p = (t1 + abs(off_u + k * oversample)) << 1;
                const double c_re = conv_func[p];
                const double c_im = conv_func[p + 1] * conv_conj;
                p = (p1 + k) << 1;
                grid[p]     += (v_re * c_re - v_im * c_im);
                grid[p + 1] += (v_im * c_re + v_re * c_im);
                sum += c_re; /* Real part only. */
            }
        }
        *norm += sum * weight_i;
    }
}

void oskar_grid_wproj_f(
        const size_t num_w_planes,
        const int* restrict support,
        const int oversample,
        const int conv_size_half,
        const float* restrict conv_func,
        const size_t num_points,
        const float* restrict uu,
        const float* restrict vv,
        const float* restrict ww,
        const float* restrict vis,
        const float* restrict weight,
        const float cell_size_rad,
        const float w_scale,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        float* restrict grid)
{
    size_t i;
    const size_t kernel_dim = conv_size_half * conv_size_half;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        /*
        //if ((i%130816)!=120776) continue;
        if ((i%130816)!=109571) continue;
        //if ((i%130816)!=108953 && (i%130816)!=109367 && (i%130816)!=109368 && (i%130816)!=109574 && (i%130816)!=109571) continue;
        if (i>=130816*90+120778) continue;
        if (i<130816*85+120778) continue;
        */
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const float ww_i = ww[i];
        const float conv_conj = (ww_i > 0.0f) ? -1.0f : 1.0f;
        const size_t grid_w = (size_t)roundf(sqrtf(fabsf(ww_i * w_scale)));
        //const size_t grid_w = 2;
        const int grid_u = (int)roundf(pos_u) + grid_centre;
        const int grid_v = (int)roundf(pos_v) + grid_centre;

        //printf("pos u %f pos v %f\n", pos_u, pos_v);
        //printf("grid u %d grid v %d\n", grid_u, grid_v);

        //if (grid_u==9228 && grid_v==9257) printf("bl: %d\n", i%130816);
        //if (grid_u>9228-5 &&grid_u<9228+5 && grid_v>9257-5 && grid_v<9257+5) printf("bl: %d\n", i%130816);
        /* Get visibility data. */
        // hard code weight value to 1.0
        const float weight_i = 1.0;
        # if FAKE_VIS_VALUES==1
        const float v_re = 1.0;
        const float v_im = 0.0;
        # else
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];
        # endif

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)roundf((roundf(pos_u) - pos_u) * oversample);
        const int off_v = (int)roundf((roundf(pos_v) - pos_v) * oversample);
       
        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];

        //const int w_support = 4;
        const size_t kernel_start = grid_w < num_w_planes ?
                grid_w * kernel_dim : (num_w_planes - 1) * kernel_dim;

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            *num_skipped += 1;
            continue;
        }


        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j)
        {
            size_t p1, t1;
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            t1 = abs(off_v + j * oversample);
            t1 *= conv_size_half;
            t1 += kernel_start;
            for (k = -w_support; k <= w_support; ++k)
            {
                size_t p = (t1 + abs(off_u + k * oversample)) << 1;
                # if FAKE_KERNEL_VALUES==1
                const float c_re = 1.0;
                const float c_im = 0.0;
                # else
                const float c_re = conv_func[p];
                const float c_im = conv_func[p + 1] * conv_conj;
                # endif
                p = (p1 + k) << 1;
                grid[p]     += (v_re * c_re - v_im * c_im);
                grid[p + 1] += (v_im * c_re + v_re * c_im);
                sum += c_re; /* Real part only. */
            }
        }
        *norm += sum * weight_i;
    }
}

void oskar_grid_wproj_f_diagnostics(
        const size_t num_w_planes,
        const int* restrict support,
        const int oversample,
        const int conv_size_half,
        const float* restrict conv_func,
        const size_t num_points,
        const float* restrict uu,
        const float* restrict vv,
        const float* restrict ww,
        const float* restrict vis,
        const float* restrict weight,
        const float cell_size_rad,
        const float w_scale,
        const int grid_size,
        size_t* restrict num_skipped,
        double* restrict norm,
        float* restrict grid,
        long long int* num_vis_per_wsupport,
        long long int* num_vis_per_wplane,
        long long int* num_vis_per_grid,
        float* avg_wsupport_per_grid,
        float* avg_wplane_per_grid)
{
    size_t i;
    const size_t kernel_dim = conv_size_half * conv_size_half;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    /* Loop over visibilities. */
    *num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const float ww_i = ww[i];
        const float conv_conj = (ww_i > 0.0f) ? -1.0f : 1.0f;
        const size_t grid_w = (size_t)roundf(sqrtf(fabsf(ww_i * w_scale)));
        //const size_t grid_w = 2;
        const int grid_u = (int)roundf(pos_u) + grid_centre;
        const int grid_v = (int)roundf(pos_v) + grid_centre;

        //printf("ww_i %f, grid_w %ld\n", ww_i, (long int) grid_w);

        //printf("pos u %f pos v %f\n", pos_u, pos_v);
        //printf("grid u %d grid v %d\n", grid_u, grid_v);

        //if (grid_u==9228 && grid_v==9257) printf("bl: %d\n", i%130816);
        //if (grid_u>9228-5 &&grid_u<9228+5 && grid_v>9257-5 && grid_v<9257+5) printf("bl: %d\n", i%130816);
        /* Get visibility data. */
        // hard code weight value to 1.0
        const float weight_i = 1.0;
        # if FAKE_VIS_VALUES==1
        const float v_re = 1.0;
        const float v_im = 0.0;
        # else
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];
        # endif

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)roundf((roundf(pos_u) - pos_u) * oversample);
        const int off_v = (int)roundf((roundf(pos_v) - pos_v) * oversample);
       
        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];

        num_vis_per_wsupport[w_support]+=1;
        num_vis_per_wplane[grid_w]+=1;
        //if (w_support!=4 && w_support!=72) printf("%d\n", w_support);
        //num_vis_per_wsupport[w_support]=5;

        //const int w_support = 4;
        const size_t kernel_start = grid_w < num_w_planes ?
                grid_w * kernel_dim : (num_w_planes - 1) * kernel_dim;

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            *num_skipped += 1;
            continue;
        }


        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j)
        {
            size_t p1, t1;
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            t1 = abs(off_v + j * oversample);
            t1 *= conv_size_half;
            t1 += kernel_start;
            for (k = -w_support; k <= w_support; ++k)
            {
                size_t p = (t1 + abs(off_u + k * oversample)) << 1;
                # if FAKE_KERNEL_VALUES==1
                const float c_re = 1.0;
                const float c_im = 0.0;
                # else
                const float c_re = conv_func[p];
                const float c_im = conv_func[p + 1] * conv_conj;
                # endif
                p = (p1 + k) << 1;
                grid[p]     += (v_re * c_re - v_im * c_im);
                grid[p + 1] += (v_im * c_re + v_re * c_im);
                num_vis_per_grid[p>>1]++;
                avg_wsupport_per_grid[p>>1] += w_support;
                avg_wsupport_per_grid[p>>1] += grid_w;
                //if (w_support>avg_wsupport_per_grid[p>>1]){
                    //avg_wsupport_per_grid[p>>1] = w_support;
                //}

                sum += c_re; /* Real part only. */
            }
        }
        *norm += sum * weight_i;
    }
}



#ifdef __cplusplus
}
#endif
