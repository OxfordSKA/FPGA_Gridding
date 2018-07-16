#pragma OPENCL EXTENSION cl_altera_channels : enable  //you need this


#define MAX_W_SUPPORT 72

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
    int compact_start_index, index_in_compact_kernel, grid_local_u, grid_local_v;
    int k_stride, off_v, oversample;
    int jstart, jend, kstart, kend;
    int wsupport;
    float conv_mul;
    float2 val;
    int load_new_conv_kernel;
    int conv_kernel_size;
};

struct ChDataVisLarge {
    int compact_start_index, grid_local_u, grid_local_v;
    int k_stride, off_v, oversample;
    int jstart, jend, kstart, kend;
    int wsupport;
    float conv_mul;
    float2 val;
};

channel struct ChDataConvEngConfig chConvEngConfig __attribute__((depth(1)));
channel struct ChDataTileConfig chTileConfig __attribute__((depth(4)));
channel struct ChDataVis chVis __attribute__((depth(16)));
channel uchar chConvEngFinished __attribute__((depth(1)));

channel struct ChDataConvEngConfig chConvEngConfigLarge __attribute__((depth(1)));
channel struct ChDataTileConfig chTileConfigLarge __attribute__((depth(4)));
channel struct ChDataVisLarge chVisLarge __attribute__((depth(16)));
channel uchar chConvEngFinishedLarge __attribute__((depth(1)));


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
       __global int* restrict tileHasLargeWSupport,
       int numTilesWithLargeWSupport,
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
    #define TILE_HAS_LARGE_W_SUPPORT(uu, vv)  tileHasLargeWSupport[( (uu) + (vv)*numTiles_u )]

    #define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
    #define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

    struct ChDataConvEngConfig conv_eng_config;
    conv_eng_config.compact_wkernel = compact_wkernel;
    conv_eng_config.tileHeight = tileHeight;
    conv_eng_config.tileWidth = tileWidth;
    conv_eng_config.trimmed_grid_size = trimmed_grid_size;
    write_channel_intel(chConvEngConfig, conv_eng_config);
    write_channel_intel(chConvEngConfigLarge, conv_eng_config);
    //printf("sent conv eng config\n");

    int nTilesSmall=nTiles-numTilesWithLargeWSupport;
    int nTilesLarge=numTilesWithLargeWSupport;
    int countTilesSmall=0, countTilesLarge=0;
    //printf("small: %d, large: %d, total: %d\n", nTilesSmall, nTilesLarge, nTiles);
    
    // Loop over the Tiles.
    for (int tile=0; tile<nTiles; tile++)
    {
        //printf("beginning of tile\n");
		// Get some information about this Tile.
		int pu = workQueue_pu[tile];
		int pv = workQueue_pv[tile];

        const int off = OFFSETS_IN_TILES(pu, pv);
		const int num_tile_vis = NUM_POINTS_IN_TILES(pu,pv);

        int tileTopLeft_u = pu*tileWidth;
        int tileTopLeft_v = pv*tileHeight;
        int tileOffset = tileTopLeft_v*trimmed_grid_size + tileTopLeft_u;

        struct ChDataTileConfig tile_config;
        tile_config.grid_pointer = (__global float2 *restrict)&grid[tileOffset*2];
        tile_config.num_tile_vis = num_tile_vis;
        tile_config.is_final = 0;

        if (!(TILE_HAS_LARGE_W_SUPPORT(pu,pv))){

            // SMALL KERNELS

            //printf("small count=%d\n", countTilesSmall);
            if (countTilesSmall==nTilesSmall-1) {
                tile_config.is_final = 1;
            } else if (num_tile_vis==0) {
                countTilesSmall++;
                continue;
            }
            countTilesSmall++;
            write_channel_intel(chTileConfig, tile_config);
            //printf("sent small tile config %d num vis %d is final %d\n", tile, tile_config.num_tile_vis, tile_config.is_final);

            int grid_w_prev = -1; 
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

                struct ChDataVis vis;

                vis.conv_kernel_size = (oversample/2+1)*(2*wsupport+1)*(2*wsupport+1)*(oversample/2+2) + (2*wsupport)*(2*wsupport+2);

                int conv_len = 2*wsupport + 1;
                int width = (oversample/2 * conv_len + 1) * conv_len;
                vis.conv_kernel_size = width*(oversample/2+1);
                int mid = (abs(off_u) + 1) * width - 1 - wsupport;
                vis.index_in_compact_kernel = mid; 
                vis.compact_start_index = compact_start; 
                vis.grid_local_u = grid_local_u;
                vis.grid_local_v = grid_local_v;
                vis.k_stride = off_u >= 0 ? 1 : -1;
                vis.kstart = kstart;
                vis.kend = kend;
                vis.off_v = off_v;
                vis.jstart = jstart;
                vis.jend = jend;
                vis.oversample = oversample;

                vis.wsupport = wsupport;
                vis.conv_mul = (float) conv_mul; 
                vis.val = val;
                vis.load_new_conv_kernel = (grid_w==grid_w_prev ? 0 : 1);
                grid_w_prev = grid_w;

                //printf("sent vis at kstart: %d\n", vis.kstart);
                write_channel_intel(chVis, vis);
                
            } // END loop over vis in tile
            //printf("finished tile %d\n", countTilesSmall);
        } else {

            // LARGE KERNELS

            //printf("large count=%d\n", countTilesLarge);
            if (countTilesLarge==nTilesLarge-1) {
                tile_config.is_final = 1;
            } else if (num_tile_vis==0) {
                countTilesLarge++;
                continue;
            }
            countTilesLarge++;
			write_channel_intel(chTileConfigLarge, tile_config);
			//printf("sent large tile config %d num vis %d is final %d\n", tile, tile_config.num_tile_vis, tile_config.is_final);

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

				struct ChDataVisLarge visLarge;

                int conv_len = 2*wsupport + 1;
                int width = (oversample/2 * conv_len + 1) * conv_len;
                int mid = (abs(off_u) + 1) * width - 1 - wsupport;
                visLarge.compact_start_index = compact_start + mid; 
                visLarge.grid_local_u = grid_local_u;
                visLarge.grid_local_v = grid_local_v;
                visLarge.k_stride = off_u >= 0 ? 1 : -1;
                visLarge.off_v = off_v;

                int reverse_k = (off_u < 0);
                //visLarge.kstart = kstart;
                //visLarge.kend = kend;

                if (reverse_k){
                    visLarge.kstart = MAX_W_SUPPORT-kend + reverse_k*(2*MAX_W_SUPPORT+1);
                } else {
                    visLarge.kstart = kstart + MAX_W_SUPPORT;
                }
                visLarge.kend = visLarge.kstart + (kend-kstart);


                visLarge.jstart = jstart;
                visLarge.jend = jend;
                visLarge.oversample = oversample;

                visLarge.wsupport = wsupport;
                visLarge.conv_mul = (float) conv_mul; 
                visLarge.val = val;

                write_channel_intel(chVisLarge, visLarge);

			} // END loop over vis in tile
        }

    } // END loop over tiles

    uchar finished = read_channel_intel(chConvEngFinished);
    finished = read_channel_intel(chConvEngFinishedLarge);
    //printf("finished: %u\n", finished);


#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES

#endif
}

__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(1)))
__kernel void convEngSmall()
{

#if 1
    double norm = 0.0;
    int num_skipped = 0;
    
    bool final_iter = 0;
    #define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
    #define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

    #define MAX_TILE_WIDTH 64
    #define MAX_TILE_HEIGHT 64
            
    // initial version -- (oversample/2 + wsupport*oversample + 1)^2
    //#define MAX_CONV_KERNEL_SIZE 84681

    // further compacted version -- (oversample/2+1)*(2*wsupport+1)^2 * (oversample/2 + 2) + (2*wsupport)*(2*wsupport+2)
    //#define MAX_CONV_KERNEL_SIZE 21852 // wsupportMax = 20
    //#define MAX_CONV_KERNEL_SIZE 33812 // wsupportMax = 25
    //#define MAX_CONV_KERNEL_SIZE 48372 // wsupportMax = 30
    //#define MAX_CONV_KERNEL_SIZE 85292 // wsupportMax = 40

    // Fred kernels -- oversample/2 * (conv_len + 1) * conv_len * (oversample/2+1)
    // fix -- either this or above is not right
    #define MAX_CONV_KERNEL_SIZE 22692 // wsupportMax = 30
    
    //#define MAX_CONV_KERNEL_SIZE 85292 // wsupportMax = 40


    float2 grid_local[MAX_TILE_WIDTH][MAX_TILE_HEIGHT];
    float2 conv_kernel_local[MAX_CONV_KERNEL_SIZE];

    struct ChDataConvEngConfig conv_eng_config;
    conv_eng_config = read_channel_intel(chConvEngConfig);
    int tileHeight = conv_eng_config.tileHeight;
    int tileWidth = conv_eng_config.tileWidth;
    if (conv_eng_config.tileHeight > 0) final_iter = 1;
    int trimmed_grid_size = conv_eng_config.trimmed_grid_size;
    __global float2* restrict compact_wkernel = (__global float2* restrict)conv_eng_config.compact_wkernel;

    
    int count=0;
    while(1){
        // Iterate over tiles
        struct ChDataTileConfig tile_config;
        //printf("in autorun final_iter: %d\n", final_iter);
        if (final_iter ==1){
            tile_config = read_channel_intel(chTileConfig); 
        }
        //printf("small received tile config. numVis: %d\n", tile_config.num_tile_vis);
        // Copy global grid to grid_local
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                // copy pointer to local first
                grid_local[y][x] = tile_config.grid_pointer[y*trimmed_grid_size + x];
            }
        }

        // Convolve this point.
        double sum = 0.0;

        // Loop over visibilities in the Tile.
        for (int i = 0; i < tile_config.num_tile_vis; i++)
        {
            struct ChDataVis vis;
            vis = read_channel_intel(chVis);
            if (vis.load_new_conv_kernel){
                for (int i=0; i<vis.conv_kernel_size; i++){
                    conv_kernel_local[i] = compact_wkernel[vis.compact_start_index + i];
                }
            }

            #pragma ivdep
            for (int j = vis.jstart; j <= vis.jend; ++j)
            {
                  // Compiler assumes there's a dependency but there isn't
                  // as threads don't access overlapping grid regions.
                  // So we have to tell the compiler explicitly to vectorise.
                #pragma ivdep
                for (int k = vis.kstart; k <= vis.kend; ++k)
                {
                    int p = vis.index_in_compact_kernel - abs(vis.off_v + j*vis.oversample)*(2*vis.wsupport+1) +
                            k*vis.k_stride;

                    float2 c = conv_kernel_local[p];
                    //float2 c = 0;
                    c.y *= vis.conv_mul;

                    // Real part only.
                    sum += c.x;

                    grid_local[vis.grid_local_v + j][vis.grid_local_u + k] +=
                        (float2) ((vis.val.x * c.x - vis.val.y * c.y), (vis.val.y * c.x + vis.val.x * c.y));
                }
            }
            //norm += sum * w;

            
        } // END loop over vis in tile

        //printf("finished tile in small autorun kernel %d\n", tile_config.is_final);

        // put the grid back
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                tile_config.grid_pointer[y*trimmed_grid_size + x] = grid_local[y][x];
            }
        }

        if (tile_config.is_final==1) {
            write_channel_intel(chConvEngFinished, tile_config.is_final);
            break;
        }
    } // END loop over tiles


#endif
}

__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(1)))
__kernel void convEngLarge()
{
#if 1
    double norm = 0.0;
    int num_skipped = 0;
    
    bool final_iter = 0;
    #define MAX( A, B ) ( (A) > (B) ? (A) : (B) )
    #define MIN( A, B ) ( (A) < (B) ? (A) : (B) )

    #define MAX_TILE_WIDTH 64
    #define MAX_TILE_HEIGHT 64

    float2 grid_local[MAX_TILE_WIDTH][MAX_TILE_HEIGHT];

    struct ChDataConvEngConfig conv_eng_config;
    conv_eng_config = read_channel_intel(chConvEngConfigLarge);
    int tileHeight = conv_eng_config.tileHeight;
    int tileWidth = conv_eng_config.tileWidth;
    if (conv_eng_config.tileHeight > 0) final_iter = 1;
    int trimmed_grid_size = conv_eng_config.trimmed_grid_size;
    __global float2* restrict compact_wkernel = (__global float2* restrict)conv_eng_config.compact_wkernel;

    int k_indices_local[2*(2*MAX_W_SUPPORT+1)];
    int numIndices = 2*MAX_W_SUPPORT+1; 
    for (int i=0; i<numIndices; i++){
        k_indices_local[i] = -MAX_W_SUPPORT + i;
    }
    for (int i=0; i<numIndices; i++){
        k_indices_local[i+numIndices] =  MAX_W_SUPPORT - i;
    }

    
    int count=0;
    while(1){
        // Iterate over tiles
        struct ChDataTileConfig tile_config;
        //printf("in autorun final_iter: %d\n", final_iter);
        if (final_iter ==1){
            tile_config = read_channel_intel(chTileConfigLarge); 
        }
        //printf(" large received tile config. numVis: %d\n", tile_config.num_tile_vis);
        // Copy global grid to grid_local
        __global float2 *restrict grid_pointer;
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                // copy pointer to local first
                grid_pointer = (__global float2 *restrict)&tile_config.grid_pointer[y*trimmed_grid_size + x];
                grid_local[y][x] = *grid_pointer;
            }
        }

        // Convolve this point.
        double sum = 0.0;

        // Loop over visibilities in the Tile.
        for (int i = 0; i < tile_config.num_tile_vis; i++)
        {
            struct ChDataVisLarge vis;
            vis = read_channel_intel(chVisLarge);

            #pragma ivdep
            for (int j = vis.jstart; j <= vis.jend; ++j)
            {
                  // Compiler assumes there's a dependency but there isn't
                  // as threads don't access overlapping grid regions.
                  // So we have to tell the compiler explicitly to vectorise.
                #pragma ivdep
                for (int kindex = vis.kstart; kindex <= vis.kend; ++kindex)
                {
                    int k = k_indices_local[kindex];
                    int p = vis.compact_start_index - abs(vis.off_v + j*vis.oversample)*(2*vis.wsupport+1) +
                            k*vis.k_stride;

                    float2 c = compact_wkernel[p];
                    c.y *= vis.conv_mul;

                    // Real part only.
                    sum += c.x;

                    grid_local[vis.grid_local_v + j][vis.grid_local_u + k] +=
                        (float2) ((vis.val.x * c.x - vis.val.y * c.y), (vis.val.y * c.x + vis.val.x * c.y));
                }
            }
            //norm += sum * w;

            
        } // END loop over vis in tile

        //printf("finished tile in large autorun kernel %d\n", tile_config.is_final);

        // put the grid back
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                grid_pointer = (__global float2 *restrict)&tile_config.grid_pointer[y*trimmed_grid_size + x];
                *grid_pointer = grid_local[y][x];
            }
        }

        if (tile_config.is_final==1) {
            write_channel_intel(chConvEngFinishedLarge, tile_config.is_final);
            break;
        }
    } // END loop over tiles


#endif
}
