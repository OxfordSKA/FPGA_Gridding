#pragma OPENCL EXTENSION cl_altera_channels : enable  //you need this

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
    int u_fac, v_fac;
    int jstart, jend, kstart, kend;
    int wsupport;
    float conv_mul;
    float2 val;
    int load_new_conv_kernel;
    int conv_kernel_size;
};

channel struct ChDataConvEngConfig chConvEngConfig __attribute__((depth(1)));
channel struct ChDataTileConfig chTileConfig __attribute__((depth(4)));
channel struct ChDataVis chVis __attribute__((depth(16)));
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
        //printf("beginning of tile\n");
		// Get some information about this Tile.
		int pu = workQueue_pu[tile];
		int pv = workQueue_pv[tile];

        const int off = OFFSETS_IN_TILES(pu, pv);
		const int num_tile_vis = NUM_POINTS_IN_TILES(pu,pv);

        if (num_tile_vis==0 && (tile<nTiles-1)) continue;

        int tileTopLeft_u = pu*tileWidth;
        int tileTopLeft_v = pv*tileHeight;
        int tileOffset = tileTopLeft_v*trimmed_grid_size + tileTopLeft_u;

        struct ChDataTileConfig tile_config;
        tile_config.grid_pointer = (__global float2 *restrict)&grid[tileOffset*2];
        tile_config.num_tile_vis = num_tile_vis;
        tile_config.is_final = 0;
        if (tile==nTiles-1) tile_config.is_final = 1;
        write_channel_intel(chTileConfig, tile_config);
        //printf("sent tile config %d num vis %d is final %d\n", tile, tile_config.num_tile_vis, tile_config.is_final);

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

            int u_fac = 0, v_fac = 0;
            if (off_u >= 0) u_fac = 1;
            if (off_u < 0) u_fac = -1;
            if (off_v >= 0) v_fac = 1;
            if (off_v < 0) v_fac = -1;

            struct ChDataVis vis;

            vis.index_in_compact_kernel = abs(off_v)*(oversample/2 + 1)*(2*wsupport + 1)*(2*wsupport + 1)  + abs(off_u)*(2*wsupport + 1)*(2*wsupport + 1)  + wsupport*(2*wsupport + 1) + wsupport; 
            vis.compact_start_index = compact_start;
            vis.conv_kernel_size = (oversample-1)*(2*wsupport+1)*(2*wsupport+1)*(oversample/2+2) + (2*wsupport)*(2*wsupport+2);

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
            vis.load_new_conv_kernel = (grid_w==grid_w_prev ? 0 : 1);
            grid_w_prev = grid_w;

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
__attribute__((autorun))
__attribute__((num_compute_units(1)))
__kernel void convEng()
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

    // further compacted version -- (oversample-1)*(2*wsupport+1)^2 * (oversample/2 + 2) + (2*wsupport)*(2*wsupport+2)
    #define MAX_CONV_KERNEL_SIZE 273324

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
        //printf("received tile config. numVis: %d\n", tile_config.num_tile_vis);
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
                    int p = vis.index_in_compact_kernel + j*vis.v_fac*(2*vis.wsupport+1) + k*vis.u_fac;

                    float2 c = conv_kernel_local[p];
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
            //norm += sum * w;

            
        } // END loop over vis in tile

        //printf("finished tile in autorun kernel\n");

        // put the grid back
        for (int y=0; y<tileHeight; y++){
            for (int x=0; x<tileWidth; x++){
                //! fix -- is this cast still required?
                tile_config.grid_pointer[y*trimmed_grid_size + x] = grid_local[y][x];
            }
        }

        if (tile_config.is_final==1) {
            write_channel_intel(chConvEngFinished, tile_config.is_final);
        }
    } // END loop over tiles


#endif
}

__attribute__((max_global_work_dim(0)))
__kernel void process_large_wsupport(
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
    int i;
    const int kernel_dim = conv_size_half * conv_size_half;
    const int grid_centre = grid_size / 2;
    const float grid_scale = grid_size * cell_size_rad;

    float norm=0;
    float2 grid_local[145][145]; 

    /* Loop over visibilities. */
    int num_skipped = 0;
    for (i = 0; i < num_points; ++i)
    {
        double sum = 0.0;
        int j, k;

        /* Convert UV coordinates to grid coordinates. */
        const float pos_u = -uu[i] * grid_scale;
        const float pos_v = vv[i] * grid_scale;
        const float ww_i = ww[i];
        const float conv_conj = (ww_i > 0.0f) ? -1.0f : 1.0f;
        const int grid_w = (int)round(sqrt(fabs(ww_i * w_scale)));
        const int grid_u = (int)round(pos_u) + grid_centre - grid_topLeft_x;
        const int grid_v = (int)round(pos_v) + grid_centre - grid_topLeft_y;

        /* Get visibility data. */
        // hard code weight value to 1.0
        const float weight_i = 1.0;
        const float v_re = weight_i * vis[2 * i];
        const float v_im = weight_i * vis[2 * i + 1];

        /* Scaled distance from nearest grid point. */
        const int off_u = (int)round((round(pos_u) - pos_u) * oversample);
        const int off_v = (int)round((round(pos_v) - pos_v) * oversample);

        /* Get kernel support size and start offset. */
        const int w_support = grid_w < num_w_planes ?
                support[grid_w] : support[num_w_planes - 1];

        const int kernel_start = grid_w < num_w_planes ?
                grid_w * kernel_dim : (num_w_planes - 1) * kernel_dim;

        __global float2 *restrict grid_pointer;
        grid_pointer = (__global float2 *restrict)grid;
        //local grid
        for (char y= -w_support; y <= w_support; y++) {
            for (char x = -w_support; x <= w_support; x++) {
                grid_pointer = (__global float2 * restrict)&grid[((grid_v+y)*trimmed_grid_size + (grid_u+x))*2];
                grid_local[y+w_support][x+w_support] = *grid_pointer;
            }
        }

        /* Convolve this point onto the grid. */
        for (j = -w_support; j <= w_support; ++j) // maximum -70,70
        {
            int p1, t1;
            p1 = grid_v + j; //corner of the convolution kernel 
            p1 *= trimmed_grid_size; /* Tested to avoid int overflow. */
            p1 += grid_u;
            t1 = abs(off_v + j * oversample);
            t1 *= conv_size_half;
            t1 += kernel_start;
            for (k = -w_support; k <= w_support; ++k)  //maximum -70, 70
            {
                int p = (t1 + abs(off_u + k * oversample)) << 1;
                const float c_re = conv_func[p];
                const float c_im = conv_func[p + 1] * conv_conj;
                
                p = (p1 + k) << 1;
                grid_local[j+w_support][k+w_support] += (float2)((v_re * c_re - v_im * c_im), (v_im * c_re + v_re * c_im));
                sum += c_re; /* Real part only. */
            }
        }

        // put the grid back
        for (char y= -w_support; y <= w_support; y++) {
            for (char x = -w_support; x <= w_support; x++) {
                grid_pointer = (__global float2 * restrict)&grid[((grid_v+y)*trimmed_grid_size + (grid_u+x))*2];
                *grid_pointer = grid_local[y+w_support][x+w_support];
            }
        }
    }
}

