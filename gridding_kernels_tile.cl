#pragma OPENCL EXTENSION cl_altera_channels : enable  //you need this


#define CONV_UNROLL 4

struct ChDataConvEngConfig{       // Fix -- might want to send these through per vis, instead of grid_w
    __global const float2* restrict compact_wkernel;
    int tileHeight, tileWidth;
    int trimmed_grid_size;
};

struct ChDataTileConfig {
    __global float2* restrict grid_pointer;
    int num_tile_vis;
    uchar is_final;
};

struct ChDataVis {
    int compact_start_index, grid_local_u, grid_local_v;
    int u_fac, v_fac;
    int jstart, jend, kstart, kend;
    int wsupport;
    float conv_mul;
    float2 val;
};

channel struct ChDataConvEngConfig chConvEngConfig __attribute__((depth(1)));
channel struct ChDataTileConfig chTileConfig __attribute__((depth(1)));
channel struct ChDataVis chVis __attribute__((depth(8)));
channel uchar chConvEngFinished __attribute__((depth(1)));

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
    //const int nTiles = 623;
    //printf("nTiles = %d\n", nTiles);
    //const int nTiles = 1;

    double norm = 0.0;
    int num_skipped = 0;

    #define NUM_POINTS_IN_TILES(uu, vv)  numPointsInTiles[( (uu) + (vv)*numTiles_u)]
    #define OFFSETS_IN_TILES(uu, vv)     offsetsPointsInTiles[( (uu) + (vv)*numTiles_u)]

    #define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
    #define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

    struct ChDataConvEngConfig conv_eng_config;
    conv_eng_config.compact_wkernel = compact_wkernel;
    conv_eng_config.tileHeight = tileHeight;
    conv_eng_config.tileWidth = tileWidth;
    conv_eng_config.trimmed_grid_size = trimmed_grid_size;
    write_channel_intel(chConvEngConfig, conv_eng_config);
    //printf("sent conv eng config\n");

    // Loop over the Tiles.
    for (int tile=0; tile<nTiles; tile++)
    {
        //printf("beginning of tile %d\n", tile);
		// Get some information about this Tile.
		int pu = workQueue_pu[tile];
		int pv = workQueue_pv[tile];

        int tileTopLeft_u = pu*tileWidth;
        int tileTopLeft_v = pv*tileHeight;
        int tileOffset = tileTopLeft_v*trimmed_grid_size + tileTopLeft_u;
		
        const int off = OFFSETS_IN_TILES(pu, pv);
		const int num_tile_vis = NUM_POINTS_IN_TILES(pu,pv);

        struct ChDataTileConfig tile_config;
        tile_config.grid_pointer = (__global float2 *restrict)&grid[tileOffset*2];
        tile_config.num_tile_vis = num_tile_vis;
        tile_config.is_final = 0;
        if (tile==nTiles-1) tile_config.is_final = 1;
        write_channel_intel(chTileConfig, tile_config);
        //printf("sent tile config %d num vis %d is final %d\n", tile, tile_config.num_tile_vis, tile_config.is_final);

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
			// Scaled distance from nearest grid point.
			const int off_u = (int)round( (round(pos_u)-pos_u) * oversample);   // \in [-oversample/2, oversample/2]
			const int off_v = (int)round( (round(pos_v)-pos_v) * oversample);    // \in [-oversample/2, oversample/2]


			const int compact_start = compact_wkernel_start[grid_w];

			int conv_mul = (ww_i > 0 ? -1 : 1);
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

            struct ChDataVis vis;
            vis.compact_start_index = compact_start + abs(off_v)*(oversample/2 + 1)*(2*wsupport + 1)*(2*wsupport + 1)  + abs(off_u)*(2*wsupport + 1)*(2*wsupport + 1)  + wsupport*(2*wsupport + 1) + wsupport; 
            vis.grid_local_u = grid_local_u;
            vis.grid_local_v = grid_local_v;
            vis.u_fac = u_fac;
            vis.v_fac = v_fac;
            vis.jstart = jstart;
            vis.jend = jend;
            vis.kstart = kstart;
            vis.kend = kend;
            vis.wsupport = wsupport;
            vis.conv_mul = (float) conv_mul; 
            vis.val = val;

            write_channel_intel(chVis, vis);
            
		} // END loop over vis in tile
        //printf("finished tile %d\n", tile);

    } // END loop over tiles

    uchar finished = read_channel_intel(chConvEngFinished);
    //printf("finished: %u\n", finished);


#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES

#endif
}

__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(1)))
__kernel void convEng()
{

#if 1
    //printf("in convEng\n");
    double norm = 0.0;
    int num_skipped = 0;
    
    bool final_iter = 0;
    #define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
    #define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

    #define MAX_TILE_WIDTH 64
    #define MAX_TILE_HEIGHT 64


    //float2 grid_local[MAX_TILE_WIDTH*MAX_TILE_HEIGHT/CONV_UNROLL][CONV_UNROLL];
    float2 grid_local[MAX_TILE_WIDTH][MAX_TILE_WIDTH];

    struct ChDataConvEngConfig conv_eng_config;
    conv_eng_config = read_channel_intel(chConvEngConfig);
    int tileHeight = conv_eng_config.tileHeight;
    int tileWidth = conv_eng_config.tileWidth;
    if (conv_eng_config.tileHeight > 0) final_iter = 1;
    int trimmed_grid_size = conv_eng_config.trimmed_grid_size;
    __global float2* restrict compact_wkernel = (__global float2* restrict)conv_eng_config.compact_wkernel;

    
    int count=0;
    while(1){
        //printf("tile: %d\n", count++); 
        // Iterate over tiles
        struct ChDataTileConfig tile_config;
        //printf("in autorun final_iter: %d\n", final_iter);
        if (final_iter ==1){
            tile_config = read_channel_intel(chTileConfig); 
        }
		__global float2 *restrict grid_pointer;
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                // copy pointer to local first
                grid_pointer = (__global float2 *restrict)&tile_config.grid_pointer[y*trimmed_grid_size + x];
                grid_local[y][x] = *grid_pointer;
                //grid_local[y][x] = 0;
            }
        }
/*
        //printf("received tile config. numVis: %d is final: %d\n", tile_config.num_tile_vis, tile_config.is_final);
        // Copy global grid to grid_local
        for (int yy=0; yy<tileHeight*tileWidth/CONV_UNROLL; yy++){
            #pragma unroll CONV_UNROLL
            for (int xx=0; xx<CONV_UNROLL; xx++){
                grid_local[yy][xx] = tile_config.grid_pointer[yy*CONV_UNROLL + xx];
            }
        }
*/

        // Convolve this point.
        double sum = 0.0;

        //printf("next tile %d\n", tile_config.is_final);
		// Loop over visibilities in the Tile.
        for (int i = 0; i < tile_config.num_tile_vis; i++)
        {   
            struct ChDataVis vis;
            vis = read_channel_intel(chVis);
            int wrap_off = vis.kstart&(CONV_UNROLL-1);
        

		    for (int j = vis.jstart; j <= vis.jend; ++j)
            {     
                  // Compiler assumes there's a dependency but there isn't
                  // as threads don't access overlapping grid regions.
                  // So we have to tell the compiler explicitly to vectorise.
                #pragma ivdep
                for (int k = vis.kstart; k <= vis.kend; ++k)
                {   
                    int p = vis.compact_start_index + j*vis.v_fac*(2*vis.wsupport+1) + k*vis.u_fac;
                    
                    float2 c = compact_wkernel[p];
                    //float2 c = 1;
                    c.y *= vis.conv_mul;
                    
                    // Real part only.
                    sum += c.x;
                    
                    //p = ((grid_v + j) * trimmed_grid_size) + grid_u + k;
                    //grid[2*p]     += (vis.val.x * c.x - vis.val.y * c.y);
                    //grid[2*p + 1] += (vis.val.y * c.x + vis.val.x * c.y);
                    grid_local[vis.grid_local_v + j][vis.grid_local_u + k] +=
                        (float2) ((vis.val.x * c.x - vis.val.y * c.y), (vis.val.y * c.x + vis.val.x * c.y));
                }
            }
/*
            #pragma ivdep
            for (int j = vis.jstart; j <= vis.jend; ++j)
            {
                  // Compiler assumes there's a dependency but there isn't
                  // as threads don't access overlapping grid regions.
                //int kstart = vis.kstart/CONV_UNROLL
                //int kend = vis.kend/CONV_UNROLL
                #pragma ivdep
                for (int xx = vis.kstart+vis.grid_local_u; xx <= vis.kend+vis.grid_local_u; xx+=CONV_UNROLL)
                {
                    #pragma unroll
                    for (int unroll_index = 0; unroll_index<CONV_UNROLL; unroll_index++){
                        int xx_rearranged = xx/CONV_UNROLL + 1*(unroll_index<wrap_off); 
                        xx_rearranged = xx_rearranged*CONV_UNROLL + unroll_index;

                        int k = xx_rearranged - vis.grid_local_u;
                        if (k>vis.kend) break;
                        if (k<vis.kstart) continue;
                        //printf("start: %d, end: %d unroll: %d, xx: %d, xx_r %d, k %d\n", vis.kstart, vis.kend, unroll_index, xx, xx_rearranged, k);

                        int p = vis.compact_start_index + j*vis.v_fac*(2*vis.wsupport+1) + k*vis.u_fac;
                        //float2 c = compact_wkernel[p];
                        float2 c;
                        c.x=1; c.y=1;
                        c.y *= vis.conv_mul;

                        // Real part only.
                        sum += c.x;
                        
                        int yy = vis.grid_local_v + j;
                        vis.val.x=1; vis.val.y=1;

                        grid_local[(yy*MAX_TILE_WIDTH + xx_rearranged)/CONV_UNROLL][unroll_index] +=
                            (float2) ((vis.val.x * c.x - vis.val.y * c.y), (vis.val.y * c.x + vis.val.x * c.y));
                    }
                }
            }
            //norm += sum * w;

*/
            
        } // END loop over vis in tile

        //printf("finished tile in autorun kernel\n");

		// put the grid back
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                grid_pointer = (__global float2 *restrict)&tile_config.grid_pointer[y*trimmed_grid_size + x];
                //*grid_pointer = grid_local[y][x];
                *grid_pointer = 0;
            }
        }
/*
        // put the grid back
        for (int yy=0; yy<tileHeight*tileWidth/CONV_UNROLL; yy++){
            #pragma unroll CONV_UNROLL
            for (int xx=0; xx<CONV_UNROLL; xx++){
                //tile_config.grid_pointer[yy*CONV_UNROLL + xx] = grid_local[yy][xx];
                tile_config.grid_pointer[yy*CONV_UNROLL + xx].x=0;
                tile_config.grid_pointer[yy*CONV_UNROLL + xx].y=0;
            }
        }
*/

        if (tile_config.is_final==1) {
            write_channel_intel(chConvEngFinished, tile_config.is_final);
            break;
        }
    } // END loop over tiles


#endif
}

