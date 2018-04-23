

__attribute__((max_global_work_dim(0)))
__kernel void oskar_process_all_tiles(
       int num_w_planes,
       __global const int* restrict support,
       int oversample, 
       __global const int* restrict compact_wkernel_start, 
       __global const float2* restrict compact_wkernel,
       float cell_size_rad,
       float w_scale,
       int trimmed_grid_size,
       int grid_size,
       int boxTop_u, int boxTop_v,
       int tileWidth, //
       int tileHeight,
       int numTiles_u, int numTiles_v,
       __global const int* restrict numPointsInTiles,
       __global const int* restrict offsetsPointsInTiles,
       __global float* restrict bucket_uu,
       __global float* restrict bucket_vv,
       __global float* restrict bucket_ww,
       __global float2* restrict bucket_vis,
       __global int * restrict workQueue_pu, 
       __global int * restrict workQueue_pv,
       __global float* restrict grid
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


#define MAX_TILE_WIDTH 64
#define MAX_TILE_HEIGHT 64

float2 grid_local[MAX_TILE_WIDTH][MAX_TILE_HEIGHT];

  // Loop over the Tiles.
#pragma omp parallel for default(shared) reduction(+:local_norm) schedule(static, 1)
  for (int tile=0; tile<nTiles; tile++)
    {
		// Get some information about this Tile.
		int pu = workQueue_pu[tile];
		int pv = workQueue_pv[tile];

        int tileTopLeft_u = pu*tileWidth;
        int tileTopLeft_v = pv*tileHeight;
        int tileOffset = tileTopLeft_v*trimmed_grid_size + tileTopLeft_u;
        __global float2 *restrict grid_pointer;
        grid_pointer = (__global float2 *restrict)grid;
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                grid_pointer = (__global float2 *restrict)&grid[(tileOffset + y*trimmed_grid_size + x)*2];
                grid_local[y][x] = *grid_pointer;
            }
        }

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

			const int grid_u = (int)round(pos_u) + g_centre - boxTop_u;
			const int grid_v = (int)round(pos_v) + g_centre - boxTop_v;
			const int grid_local_u = grid_u - tileTopLeft_u;
            const int grid_local_v = grid_v - tileTopLeft_v;
			int grid_w = (int)round(sqrt(fabs(ww_i * w_scale)));
			if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

			const int wsupport = support[grid_w];
/*
			if (grid_u + wsupport >= trimmed_grid_size || grid_u - wsupport < 0 ||
					grid_v + wsupport >= trimmed_grid_size || grid_v - wsupport < 0)
			{
				num_skipped += 1;
				continue; //TODO: Is this the best way to break the rest of loop?
			}        
*/
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

			const int tile_u[] = {pu*tileWidth, (pu+1)*tileWidth-1};
			const int tile_v[] = {pv*tileHeight, (pv+1)*tileHeight-1};

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

					//p = ((grid_v + j) * trimmed_grid_size) + grid_u + k;
					//grid[2*p]     += (val.x * c.x - val.y * c.y);
					//grid[2*p + 1] += (val.y * c.x + val.x * c.y);
                    grid_local[grid_local_v + j][grid_local_u + k] +=
                        (float2) ((val.x * c.x - val.y * c.y), (val.y * c.x + val.x * c.y));
				}
			}
			norm += sum * w;

            
		} // END loop over vis in tile

        // put the grid back
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                grid_pointer = (__global float2 *restrict)&grid[(tileOffset + y*trimmed_grid_size + x)*2];
                *grid_pointer = grid_local[y][x];
            }
        }
     

    } // END loop over tiles

  /*auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop-start;
  printf("Tile processing time: %gms\n", diff.count()*1000);
  */

#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES

#endif
}

