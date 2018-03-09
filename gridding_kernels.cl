/*__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
 
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    // Do the operation
    C[i] = A[i] + B[i];
}
*/
__attribute__((max_global_work_dim(0)))
__kernel void oskar_grid_wproj_cl(
        int num_w_planes,
        __global const int* restrict support,
        const int oversample,
        const int conv_size_half,
        __global const float* restrict conv_func,
        const int num_points,
        __global const float* restrict uu,
        __global const float* restrict vv,
        __global const float* restrict ww,
        __global const float* restrict vis,
        __global const float* restrict weight,
        const float cell_size_rad,
        const float w_scale,
        const int grid_size,
        __global float* restrict grid)
{

    printf("RUNNING KERNEL!\n");

    int i;
    const int kernel_dim = conv_size_half * conv_size_half;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    float norm=0;

    
    /* Loop over visibilities. */
    int num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
    	if ((i%1000) == 0) printf("i: %d\n", i);
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const float ww_i = ww[i];
        const float conv_conj = (ww_i > 0.0f) ? -1.0f : 1.0f;
        const int grid_w = (int)round(sqrt(fabs(ww_i * w_scale)));
        //printf("grid_w: %d, ww_i %f\n", grid_w, ww_i);
        //const int grid_w = 2;
        const int grid_u = (int)round(pos_u) + grid_centre;
        const int grid_v = (int)round(pos_v) + grid_centre;

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
        const int off_u = (int)round((round(pos_u) - pos_u) * oversample);
        const int off_v = (int)round((round(pos_v) - pos_v) * oversample);

        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];

        //const int w_support = 4;
        const int kernel_start = grid_w < num_w_planes ?
                grid_w * kernel_dim : (num_w_planes - 1) * kernel_dim;

        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= grid_size || grid_v - w_support < 0)
        {
            num_skipped += 1;
            continue;
        }


        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j)
        {
            int p1, t1;
            p1 = grid_v + j;
            p1 *= grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            t1 = abs(off_v + j * oversample);
            t1 *= conv_size_half;
            t1 += kernel_start;
            for (k = -w_support; k <= w_support; ++k)
            {
                int p = (t1 + abs(off_u + k * oversample)) << 1;
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
               // if (c_re>0) printf("grid: %.15f %.15f\n", grid[p], v_re * c_re - v_im * c_im);
                sum += c_re; /* Real part only. */
            }
                                                                }
        norm += sum * weight_i;
    }
}
