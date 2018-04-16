

__attribute__((max_global_work_dim(0)))
__kernel void oskar_process_all_tiles(
       int num_w_planes,
       __global const int* support,
       int oversample, 
       __global const int* compact_wkernel_start, 
       __global const float2* compact_wkernel,
       float cell_size_rad,
       float w_scale,
       int grid_size,
       int boxTop_u, int boxTop_v,
       int tileWidth,
       int tileHeight,
       int numTiles_u, int numTiles_v,
       __global const int* numPointsInTiles,
       __global const int* offsetsPointsInTiles,
       __global float* bucket_uu,
       __global float* bucket_vv,
       __global float* bucket_ww,
       __global float2* bucket_vis,
       __global int *workQueue_pu, 
       __global int *workQueue_pv,
       __global float* grid
       )
{
#if 1
  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;

  const int nTiles = numTiles_u * numTiles_v;

  double norm = 0.0;
  int num_skipped = 0;

#define NUM_POINTS_IN_TILES(uu, vv)  numPointsInTiles[( (uu) + (vv)*numTiles_u)]
#define OFFSETS_IN_TILES(uu, vv)     offsetsPointsInTiles[( (uu) + (vv)*numTiles_u)]

#define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
#define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

  //auto start = std::chrono::high_resolution_clock::now();

  // Loop over the Tiles.
#pragma omp parallel for default(shared) reduction(+:local_norm) schedule(static, 1)
  for (int tile=0; tile<nTiles; tile++)
    {

      // Get some information about this Tile.
      int pu = workQueue_pu[tile];
      int pv = workQueue_pv[tile];

      const int off = OFFSETS_IN_TILES(pu, pv);
      const int num_tile_vis = NUM_POINTS_IN_TILES(pu,pv);

      // Loop over visibilities in the Tile.
      for (int i = 0; i < num_tile_vis; i++)
    {
      // Convert UV coordinates to grid coordinates.
 float pos_u    = -bucket_uu[off+i] * scale;
      float pos_v    = bucket_vv[off+i] * scale;
      float ww_i     = bucket_ww[off+i];
      //const float w  = bucket_weight[off+i];
      const float w  = 1.0;
      float2 val     = bucket_vis[off+i];
      val.x *= w;
      val.y *= w;

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale)));
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      const int wsupport = support[grid_w];

        if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
                grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
        {
            num_skipped += 1;
            continue;
        }        


      // Scaled distance from nearest grid point.
      const int off_u = (int)round( (round(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
      const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]

      // Convolve this point.
      double sum = 0.0;

      const int compact_start = compact_wkernel_start[grid_w];

      float conv_mul = (ww_i > 0 ? -1.0f : 1.0f);
      // Now need to clamp iteration range to this tile.  Tile is 
      //          boxTop_u + pu*tileWidth, boxTop_v + (pu+1)*tileWidth - 1
      // and the same for pv.  Our grid iteration range is
      //       grid_u + k
      // for -wsupport <= k <= wsupport and 
      //       grid_v + j
      // for -wsupport <= j <= wsupport

              const int tile_u[] = {boxTop_u + pu*tileWidth, boxTop_u + (pu+1)*tileWidth-1};
      const int tile_v[] = {boxTop_v + pv*tileHeight, boxTop_v + (pv+1)*tileHeight-1};

      const int kstart = MAX(tile_u[0]-grid_u, -wsupport);
      const int kend   = MIN(tile_u[1]-grid_u,  wsupport);

      const int jstart = MAX(tile_v[0]-grid_v, -wsupport);
      const int jend   = MIN(tile_v[1]-grid_v,  wsupport);

      int u_fac = 0, v_fac = 0;
      if (off_u >= 0) u_fac = 1;
      if (off_u < 0) u_fac = -1;
      if (off_v >= 0) v_fac = 1;
      if (off_v < 0) v_fac = -1;

      for (int j = jstart; j <= jend; ++j)
        {
          // Compiler assumes there's a dependency but there isn't
          // as threads don't access overlapping grid regions.
          // So we have to tell the compiler explicitly to vectorise.
#pragma omp simd
          for (int k = kstart; k <= kend; ++k)
        {
          int p = compact_start + abs(off_v)*(oversample/2 + 1)*(2*wsupport + 1)*(2*wsupport + 1)  + abs(off_u)*(2*wsupport + 1)*(2*wsupport + 1)  + (j*v_fac + wsupport)*(2*wsupport + 1) + k*u_fac + wsupport;

          float2 c = compact_wkernel[p];
          c.y *= conv_mul;

          // Real part only.
          sum += c.x;

          p = ((grid_v + j) * grid_size) + grid_u + k;
          grid[2*p]     += (val.x * c.x - val.y * c.y);
          grid[2*p + 1] += (val.y * c.x + val.x * c.y);
        }
        }
      norm += sum * w;
    }
    }

  /*auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop-start;
  printf("Tile processing time: %gms\n", diff.count()*1000);
  */

#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES

#endif
}




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
        const int trimmed_grid_size,
        const int grid_size,
		const int grid_topLeft_x, const int grid_topLeft_y,
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
    //for (i = 0; i < num_points; ++i)
    for (i = 0; i < 10000; ++i)
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
        const int grid_u = (int)round(pos_u) + grid_centre - grid_topLeft_x;
        const int grid_v = (int)round(pos_v) + grid_centre - grid_topLeft_y;

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

        //printf("grid_u %d, grid_size %d\n", grid_u, grid_size);
        /* Catch points that would lie outside the grid. */
        if (grid_u + w_support >= trimmed_grid_size || grid_u - w_support < 0 ||
                grid_v + w_support >= trimmed_grid_size || grid_v - w_support < 0)
        {
            num_skipped += 1;
            continue;
        }

        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j)
        {
            int p1, t1;
            p1 = grid_v + j;
            p1 *= trimmed_grid_size; /* Tested to avoid int overflow. */
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
                //if (c_re>0) printf("grid: %.15f %.15f\n", grid[p], v_re * c_re - v_im * c_im);
                sum += c_re; /* Real part only. */
            }
                                                                }
        norm += sum * weight_i;
    }
}
