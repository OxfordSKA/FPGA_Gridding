// (C) 2012  John Romein/ASTRON

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#if defined __AVX__
#include <immintrin.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Common.h"
#include "Defines.h"

#if ORDER == ORDER_W_OV_OU_V_U
typedef REAL2 SupportType[W_PLANES][OVERSAMPLE_V][OVERSAMPLE_U][SUPPORT_V][SUPPORT_U];
#elif ORDER == ORDER_W_V_OV_U_OU
typedef REAL2 SupportType[W_PLANES][SUPPORT_V][OVERSAMPLE_V][SUPPORT_U][OVERSAMPLE_U];
#endif

typedef REAL2 GridType[GRID_V][GRID_U][POLARIZATIONS];
typedef REAL3 UVWtype[BASELINES][TIMESTEPS][CHANNELS];
typedef REAL2 VisibilitiesType[BASELINES][TIMESTEPS][CHANNELS][POLARIZATIONS];

unsigned nrThreads;

__device__ void atomicAdd(REAL2 *ptr, REAL2 sumXX)
{
    atomicAdd(&ptr[0].x, sumXX.x);
    atomicAdd(&ptr[0].y, sumXX.y);
}

__device__ void addSupportPixel(REAL2 &sum, REAL2 supportPixel, REAL2 vis)
{

    # if FAKE_VIS_VALUES==1
    vis.x=1.0;
    vis.y=0.0;
    # endif

    sum.x += supportPixel.x * vis.x;
    sum.y += supportPixel.x * vis.y;
    sum.x -= supportPixel.y * vis.y;
    sum.y += supportPixel.y * vis.x;
}

__shared__ int4   shared_info[TIMESTEPS][CHANNELS];
__shared__ REAL2 shared_vis[TIMESTEPS][CHANNELS][POLARIZATIONS];

__device__ void loadIntoSharedMem(const VisibilitiesType visibilities,
				  const UVWtype uvw,
				  const uint2 supportPixelsUsed[BASELINES],
                  unsigned full_grid_size)
{
    unsigned bl = blockIdx.x;

    uint2 supportSize = supportPixelsUsed[bl];

    int zCoord = roundf(uvw[bl][0][0].z);
    
    int grid_centre_u = full_grid_size/2 - TRIMMED_REGION_OFFSET_U + 1;
    int grid_centre_v = full_grid_size/2 - TRIMMED_REGION_OFFSET_V + 1;

    //for (unsigned ch = threadIdx.x; ch < CHANNELS * TIMESTEPS; ch += blockDim.x)
    for (int ch = threadIdx.x; ch < CHANNELS * TIMESTEPS; ch += blockDim.x)
    {
        REAL3   coords = uvw[bl][0][ch]; // coords = {u,v,w}

        // the kernel grid is shifted by half a kernel grid cell in relation to the global grid

        int u_int  = __float2int_rd(coords.x);
        int v_int  = __float2int_rd(coords.y);

        //printf("coords: %f %f, u_int %d v_int %d\n", coords.x, coords.y, u_int+TRIMMED_REGION_OFFSET_U+supportSize.x/2, v_int+TRIMMED_REGION_OFFSET_V+supportSize.y/2);

        //if (u_int>2160 && u_int<2166 && v_int<3202 && v_int>3195) printf("blockId: %d\n", blockIdx.x);
        //if (u_int==2164 && v_int==3198) printf("blockId: %d\n", blockIdx.x);

        coords.x += (1.0/OVERSAMPLE_U)/2.0;
        coords.y += (1.0/OVERSAMPLE_V)/2.0;
        int kernel_u_int  = __float2int_rd(coords.x);
        int kernel_v_int  = __float2int_rd(coords.y);

        REAL u_frac, v_frac;
        u_frac = (coords.x - kernel_u_int);
        v_frac = (coords.y - kernel_v_int);

        unsigned u_off, v_off;
        u_off = (unsigned)OVERSAMPLE_U*u_frac;
        v_off = (unsigned)OVERSAMPLE_V*v_frac;

        // convert oversample index to offset
	unsigned additional_kernel_offset = 0;
        u_off = (OVERSAMPLE_U-u_off)%4;
        v_off = (OVERSAMPLE_V-v_off)%4;
	if (u_off==0 && kernel_u_int==u_int){
		additional_kernel_offset = 1;
	}
	if (v_off==0 && kernel_v_int==v_int){
		additional_kernel_offset += SUPPORT_U;
	}

    u_int += grid_centre_u;
    v_int += grid_centre_v;
        
#if ORDER == ORDER_W_OV_OU_V_U
        unsigned uv_frac_w_offset = (unsigned) zCoord * SUPPORT_V * SUPPORT_U * OVERSAMPLE_V * OVERSAMPLE_U + SUPPORT_U * SUPPORT_V * (OVERSAMPLE_U * v_off + (unsigned) u_off) + additional_kernel_offset; // starting index of correct kernel
#elif ORDER == ORDER_W_V_OV_U_OU
        unsigned uv_frac_w_offset = (unsigned) zCoord * SUPPORT_V * OVERSAMPLE_V * SUPPORT_U * OVERSAMPLE_U + (unsigned) (OVERSAMPLE_V * v_frac) * SUPPORT_U * OVERSAMPLE_U + (unsigned) (OVERSAMPLE_U * u_frac);
#endif
        shared_info[0][ch] = make_int4(-u_int % supportSize.x, -v_int % supportSize.y, uv_frac_w_offset, u_int + GRID_U * v_int);
    }

    #if TIMESTEPS>1
    // more efficient to read these into shared memory 4 floats at a time rather than 2 floats at a time?
    //for (unsigned i = threadIdx.x; i < CHANNELS * TIMESTEPS * POLARIZATIONS / 2; i += blockDim.x)
    for (int i = threadIdx.x; i < CHANNELS * TIMESTEPS * POLARIZATIONS / 2; i += blockDim.x)
        ((REAL4 *) shared_vis)[i] = ((REAL4 *) visibilities[bl])[i];
    # else
    //for (unsigned i = threadIdx.x; i < CHANNELS * TIMESTEPS * POLARIZATIONS; i += blockDim.x)
    for (int i = threadIdx.x; i < CHANNELS * TIMESTEPS * POLARIZATIONS; i += blockDim.x)
        ((REAL2 *) shared_vis)[i] = ((REAL2 *) visibilities[bl])[i];
    #endif
}


__device__ void convolve(GridType grid,
			const SupportType support,
			const uint2 supportPixelsUsed[BASELINES])
{
    unsigned bl	= blockIdx.x;
    uint2 supportSize = supportPixelsUsed[bl];
//	if (bl!=400) return;

    for (int i = supportSize.x * supportSize.y - threadIdx.x - 1; i >= 0; i -= blockDim.x)
    {
        int box_u = - (i % supportSize.x);
        int box_v = - (i / supportSize.x);
        REAL2 sumXX = MAKE_REAL2(0, 0);
        unsigned grid_point = threadIdx.x;// does this cause (0,0) to be added when t=0?
        

    	//for (unsigned ch = 0; ch < CHANNELS * TIMESTEPS; ch++)
    	for (unsigned ch = 0; ch < 1 * TIMESTEPS; ch++)
        {
            // info = { x=-u offset of box from subgrids, y=-v offset of box from subgrids, 
            // z=start index of correct wkernel, w=index of vis-(2*wsupport+1)/2 in global grid 
            // (ie index of beginning of kernel for that visibility)}
        	int4 info = shared_info[0][ch]; 
        	int my_support_u = box_u + info.x;
        	int my_support_v = box_v + info.y;

        	if (my_support_u < 0)
        	    my_support_u += supportSize.x;

        	if (my_support_v < 0)
        	    my_support_v += supportSize.y;

            // thread's index in wkernel
        	unsigned index_u = my_support_u;
        	unsigned index_v = my_support_v;

#if ORDER == ORDER_W_OV_OU_V_U
    	    unsigned supportIndex = index_u + SUPPORT_U * index_v + info.z;
#elif ORDER == ORDER_W_V_OV_U_OU
    	    unsigned supportIndex = OVERSAMPLE_U * index_u + OVERSAMPLE_V * SUPPORT_U * OVERSAMPLE_U * index_v + info.z;
#endif
    	    REAL2 supportPixel;
            # if FAKE_KERNEL_VALUES==1
    	    supportPixel = MAKE_REAL2(1.0, 0.0);
            # else
            supportPixel= support[0][0][0][0][supportIndex];
            # endif

        	unsigned new_grid_point = my_support_u + GRID_U * my_support_v + info.w;

            if (new_grid_point != grid_point)
            {
                atomicAdd(&grid[0][grid_point][0], sumXX);
                sumXX = MAKE_REAL2(0, 0);
                grid_point = new_grid_point;
            }
            addSupportPixel(sumXX, supportPixel, shared_vis[0][ch][0]);
        }
        atomicAdd(&grid[0][grid_point][0], sumXX);
    }
}


//#if MODE == MODE_SIMPLE || MODE == MODE_OVERSAMPLE
#define MIN(A,B)			((A) < (B) ? (A) : (B))
#define NR_THREADS_PER_BLOCK		MIN(SUPPORT_U * SUPPORT_V, 1024)
#define MIN_BLOCKS_PER_MULTIPROCESSOR	(2048 / NR_THREADS_PER_BLOCK)
__global__ __launch_bounds__(NR_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)
//#else
//__global__ __launch_bounds__(1024, 1)
//#endif
void addToGrid(GridType grid,
	       const SupportType support,
	       const VisibilitiesType visibilities,
	       const UVWtype uvw,
	       const uint2 supportPixelsUsed[BASELINES],
           unsigned full_grid_size)
{
    //if (blockIdx.x!=120776) return;
    //if (blockIdx.x!=108953 && blockIdx.x!=109367 && blockIdx.x!=109368 && blockIdx.x!=109574 && blockIdx.x!=109571) return;
    //if (blockIdx.x!=109571) return;
    loadIntoSharedMem(visibilities, uvw, supportPixelsUsed, full_grid_size);
    __syncthreads();
    convolve(grid, support, supportPixelsUsed);
}



void initUVW(UVWtype uvw, uint2 supportPixelsUsed[BASELINES], const REAL frequencies[CHANNELS], unsigned block,
	unsigned full_grid_size, const double cell_size_rad, const double w_scale, 
    const REAL *uu, const REAL *vv, const REAL *ww, const int *supportSize)
{
    // we only process the trimmed region of the global grid which contains visibilities
    // input uvw coordinates will have their origin at the global grid centre
    //int grid_centre_u = full_grid_size/2 - TRIMMED_REGION_OFFSET_U + 1;
    //int grid_centre_v = full_grid_size/2 - TRIMMED_REGION_OFFSET_V + 1;
    printf("block: %d\n", block);

    for (unsigned bl = 0; bl < BASELINES; bl ++) {
        for (unsigned time = 0; time < TIMESTEPS; time ++) {

            // uu[TIMESTEPSxBLOCKS][BASELINES]
	        const int currentUVWIndex = (block*TIMESTEPS + time)*BASELINES + bl;

			REAL scaled_u = full_grid_size*cell_size_rad * -uu[currentUVWIndex];
			REAL scaled_v = full_grid_size*cell_size_rad * vv[currentUVWIndex];
            		REAL w = ww[currentUVWIndex];
        
			REAL scaled_w = sqrt(fabs(w * w_scale)); 
            if (time==0){
				// init support
				supportPixelsUsed[bl].x = supportSize[(int)roundf(scaled_w)]*2 + 1; 
				supportPixelsUsed[bl].y = supportSize[(int)roundf(scaled_w)]*2 + 1; 
            }

            // seperate kernels are stored for w<0 and w>0
            if (w>0) scaled_w = scaled_w + W_PLANES/2;
            else scaled_w = W_PLANES/2 - scaled_w; 

            // use only one channel
            uvw[bl][time][0] = MAKE_REAL3(
                    scaled_u - supportPixelsUsed[bl].x / 2.0f,
                    scaled_v - supportPixelsUsed[bl].y / 2.0f,
                    scaled_w
            );
        }
    }
}

REAL2 getSupportVal(const REAL *supportIn, const int conv_size_half, int ou, int ov, int u, int v, int w, int wNegative){
    int supportOffset = w * conv_size_half*conv_size_half;
    int supportIndex = supportOffset + (v*OVERSAMPLE_V+ov)*conv_size_half + u*OVERSAMPLE_U + ou;
    REAL real = supportIn[supportIndex*2];
    REAL imag = supportIn[supportIndex*2+1];
    if (!wNegative) imag *= -1.0;
    return MAKE_REAL2(real, imag);
}

void initSupport(SupportType support, const REAL* supportIn, const int* supportSize, const int conv_size_half)
{
    // SUPPORT_U, SUPPORT_V := 2*maxWSupport+1
    for (int wIndex = 0; wIndex < W_PLANES; wIndex ++){
        int w = wIndex-W_PLANES/2;
        int wNegative = w < 0 ? 1 : 0;
        w = abs(w);
        int wsupport = supportSize[w];
        int padding = SUPPORT_U - (2*wsupport+2);
        int uOut, vOut, u, v;
        for (int ov = 0; ov < OVERSAMPLE_V; ov ++){
            for (int ou = 0; ou < OVERSAMPLE_U; ou ++){
		vOut=0;
                // top two quadrants of kernel
                for (v=wsupport; v>=0; v--){
                    uOut = 0;
                    // left quadrant of kernel
                    for (u=wsupport; u>=0; u--){
                        support[wIndex][ov][ou][vOut][uOut++] = getSupportVal(supportIn, conv_size_half,
                            (OVERSAMPLE_U-ou-OVERSAMPLE_U/2), (OVERSAMPLE_V-ov-OVERSAMPLE_V/2), u, v, w, wNegative);
                    }
                    // right quadrant of kernel
                    uOut = uOut-1;
                    for (u=0; u<=wsupport+1; u++){
                        if (u==0 && ou<OVERSAMPLE_U/2) { uOut++; continue; }
			if (u==wsupport+1 && ou>0) {uOut++; continue; }
                        support[wIndex][ov][ou][vOut][uOut++] = getSupportVal(supportIn, conv_size_half,
                            ou-OVERSAMPLE_U/2, (OVERSAMPLE_V-ov-OVERSAMPLE_V/2), u, v, w, wNegative);
                    }
                    // padding to 2*MAX_W_SUPPORT+1
                    for (u=0; u<padding; u++) {
                        support[wIndex][ov][ou][vOut][uOut++]=MAKE_REAL2(0.,0.);
                    }

                    vOut++;
                }
		vOut = vOut-1;
                // bottom two quadrants of kernel
                for (v=0; v<=wsupport+1; v++){
                    if (v==0 && ov<OVERSAMPLE_V/2) { vOut++; continue; }
		    if (v==wsupport+1 && ov>0) {uOut++; continue; }
                    uOut = 0;
                    // left quadrant of kernel
                    for (u=wsupport; u>=0; u--){
                        support[wIndex][ov][ou][vOut][uOut++] = getSupportVal(supportIn, conv_size_half,
                            OVERSAMPLE_U-ou-OVERSAMPLE_U/2, ov-OVERSAMPLE_V/2, u, v, w, wNegative);
                    }
                    uOut = uOut-1;
                    // right quadrant of kernel
                    for (u=0; u<=wsupport+1; u++){
                        if (u==0 && ou<OVERSAMPLE_U/2) { uOut++; continue; }
			if (u==wsupport+1 && ou>0) {uOut++; continue; }
                        support[wIndex][ov][ou][vOut][uOut++] = getSupportVal(supportIn, conv_size_half,
                            ou-OVERSAMPLE_U/2, ov-OVERSAMPLE_V/2, u, v, w, wNegative);
                    }
                    // padding to 2*MAX_W_SUPPORT+1
                    for (u=0; u<padding; u++) {
                        support[wIndex][ov][ou][vOut][uOut++]=MAKE_REAL2(0.,0.);
                    }
                    vOut++;
                }

                // padding to 2*MAX_W_SUPPORT+1
                for (v=0; v<padding; v++) {
                    uOut = 0;
                    for (u=0; u<SUPPORT_U; u++){
                        support[wIndex][ov][ou][vOut][uOut++]=MAKE_REAL2(0.,0.);
                    }
                    vOut++;
                }
	    }
        }
    }
}

void initFrequencies(REAL frequencies[CHANNELS])
{
    for (unsigned ch = 0; ch < CHANNELS; ch ++)
        frequencies[ch] = 59908828.7353515625 + 12207.03125 * ch;
}


void initVisibilities(VisibilitiesType visibilities, const REAL *inputVis, unsigned block)
{
    // reorder input visibility data
    // visibilities[BASELINES][TIMESTEPS][CHANNELS][POLARIZATIONS]
    // inputVis[TIMESTEPSxBLOCKS][BASELINES][2] (as 1D array)
    REAL real, imag;
    REAL2 vis;
    for (int bl=0; bl<BASELINES; bl++){
        for (int t=0; t<TIMESTEPS; t++){
            // use one channel and one polarisation
            real = inputVis[ (block*TIMESTEPS+t)*BASELINES*2 + bl*2 ];
            imag = inputVis[ (block*TIMESTEPS+t)*BASELINES*2 + bl*2 + 1 ];
            vis = MAKE_REAL2(real, imag);
            visibilities[bl][t][0][0] = vis;
        }
    }
}

void printWorkLoad(uint2 supportPixelsUsed[BASELINES])
{
    unsigned long long gridPointUpdates = 0;

    for (unsigned bl = 0; bl < BASELINES; bl ++)
        gridPointUpdates += TIMESTEPS * CHANNELS * POLARIZATIONS * supportPixelsUsed[bl].x * supportPixelsUsed[bl].y;

#pragma omp critical (cout)
    std::cout << "gridPointUpdates = " << gridPointUpdates << std::endl;
}

void printGrid(const GridType grid, const char *who)
{
//std::cout << * (unsigned long long *) &grid[4095][4095][0] << " out of " << * (unsigned long long *) &grid[4095][4095][1] << " (" << 100.0 * * (unsigned long long *) &grid[4095][4095][0] / * (unsigned long long *) &grid[4095][4095][1] << "%)" << std::endl;
    unsigned count_v = 0;
    double2 sum = make_double2(0, 0);
    for (unsigned v = 0; v < GRID_V; v ++) {
        unsigned count_u = 0;

        for (unsigned u = 0; u < GRID_U; u ++) {
            if (grid[v][u][0].x != 0 || grid[v][u][0].y != 0) {
	            if (count_u ++ == 0)
	                count_v ++;

                if (count_u < 5 && count_v < 5)
#pragma omp critical (cout)
	               std::cout << who << ": (" << u << ", " << v << "): " << grid[v][u][0] << std::endl;
                sum.x += grid[v][u][0].x;
                sum.y += grid[v][u][0].y;
            }
        }
    }
#pragma omp critical (cout)
    std::cout << "sum = " << sum << std::endl;
}



void initSupportOnHostAndDevice(SharedObject<SupportType> &support, const REAL* conv_func, const int *supportSize,
        const int conv_size_half)
{
    initSupport(*support.hostPtr, conv_func, supportSize, conv_size_half);
    support.copyHostToDevice();
}

void copySupportToCube(SupportType supports, REAL* kernels_cube){
    int outIndex;
    REAL real, imag;
    for (int w=0; w<W_PLANES; w++){
        for (int ov = 0; ov < OVERSAMPLE_V; ov ++){
            for (int ou = 0; ou < OVERSAMPLE_U; ou ++){
                for (int v = 0; v < SUPPORT_V; v ++){
                    for (int u = 0; u < SUPPORT_U; u ++){
                        real = supports[w][ov][ou][v][u].x;
                        imag = supports[w][ov][ou][v][u].y;
                        // separate out oversample offset values for each kernel
                        //outIndex = w*OVERSAMPLE_V*OVERSAMPLE_U*SUPPORT_V*SUPPORT_U +
                         //           ov*OVERSAMPLE_U*SUPPORT_V*SUPPORT_U + ou*SUPPORT_V*SUPPORT_U +
                          //          v*SUPPORT_U + u;

                        // put all oversample offset values in same kernel
                        outIndex =  w*OVERSAMPLE_V*OVERSAMPLE_U*SUPPORT_V*SUPPORT_U + 
                                (v*OVERSAMPLE_V+ov)*SUPPORT_U*OVERSAMPLE_U + u*OVERSAMPLE_U+ou;
                        kernels_cube[2*outIndex] = real;
                        kernels_cube[2*outIndex + 1] = imag;
                    }
                }
            }
        }
    }
}


void oskar_grid_wproj_gpu(const int num_w_planes, const int* supportSize,
        const int oversample, const int conv_size_half,
        const REAL* conv_func, const int num_vis,
        const REAL* uu, 	// uu[TIMESTEPSxBLOCKS][BASELINES]
	const REAL* vv,
        const REAL* ww, 
	const REAL* vis, 	// vis[TIMESTEPSxBLOCKS][BASELINES][2] 
        const REAL* weight, const double cell_size_rad,
        const double w_scale, const int grid_size, size_t* num_skipped,
        double* norm, REAL* gridOut)
{
    int device = 0;

    checkCudaCall(cudaSetDevice(device));
    checkCudaCall(cudaSetDeviceFlags(cudaDeviceMapHost));

    SharedObject<GridType> grids[STREAMS];

    for (unsigned stream = 0; stream < STREAMS; stream ++)
        checkCudaCall(cudaMemset(grids[stream].devPtr, 0, sizeof(GridType)));

    SharedObject<SupportType> supports[STREAMS];


    // do we need to copies of support? should be read only
    for (unsigned stream = 0; stream < STREAMS; stream ++)
        initSupportOnHostAndDevice(supports[stream], conv_func, supportSize, conv_size_half);

    // for debugging
    //copySupportToCube(*supports[0].hostPtr, kernels_cube);

    REAL frequencies[CHANNELS];
    initFrequencies(frequencies);
    cudaFuncSetCacheConfig(addToGrid, cudaFuncCachePreferShared);

    unsigned nrThreads = NR_THREADS_PER_BLOCK;
    double start = getTime();

#if defined MAP_OBJECTS
    MappedObject<uint2 [BASELINES]> supportPixelsUsed[STREAMS];
    MappedObject<UVWtype> uvw[STREAMS];
    MappedObject<VisibilitiesType> visibilities[STREAMS];
#else
    SharedObject<uint2 [BASELINES]> supportPixelsUsed[STREAMS];
    SharedObject<UVWtype> uvw[STREAMS];
    SharedObject<VisibilitiesType> visibilities[STREAMS];
#endif

    Stream streams[STREAMS];

    printf("num w planes: %d\n", num_w_planes);
    
    printf("timesteps %d blocks %d streams %d bl %d channels %d\n", TIMESTEPS, BLOCKS, STREAMS, BASELINES, CHANNELS);

//#pragma omp critical (cout)
    std::cout << "using " << nrThreads << /*'/' << bestNrThreads <<*/ " threads" << std::endl;
    for (unsigned block = 0; block < BLOCKS; block += STREAMS)
    //for (unsigned block = 86; block < BLOCKS; block += STREAMS)
    {
        for (unsigned stream = 0; stream < STREAMS; stream++)
        {
	        initUVW(*uvw[stream].hostPtr, *supportPixelsUsed[stream].hostPtr, frequencies, block + stream, 
			grid_size, cell_size_rad, w_scale, uu, vv, ww, supportSize);
    	    initVisibilities(*visibilities[stream].hostPtr, vis, block + stream);
        }

    checkCudaCall(cudaDeviceSynchronize());
        for (unsigned stream = 0; stream < STREAMS; stream++)
        {
        	visibilities[stream].copyHostToDevice(streams[stream]);
        	uvw[stream].copyHostToDevice(streams[stream]);
        	supportPixelsUsed[stream].copyHostToDevice(streams[stream]);
        }

    checkCudaCall(cudaDeviceSynchronize());
        for (unsigned stream = 0; stream < STREAMS; stream++)
        {
	        printWorkLoad(*supportPixelsUsed[stream].hostPtr);
        	addToGrid<<<BASELINES, nrThreads, 0, streams[stream]>>>(
                    *grids[stream].devPtr, *supports[stream].devPtr,
                    *visibilities[stream].devPtr,
                    *uvw[stream].devPtr, *supportPixelsUsed[stream].devPtr,
                    grid_size);
    	   checkCudaCall(cudaGetLastError());
        }
    checkCudaCall(cudaDeviceSynchronize());
    }
    checkCudaCall(cudaDeviceSynchronize());

    Event startCopy, finishedCopy;
    startCopy.record();
    grids[0].copyDeviceToHost();
    finishedCopy.record();
    finishedCopy.synchronize();
    int gridOutOffset = TRIMMED_REGION_OFFSET_V*grid_size*2 + TRIMMED_REGION_OFFSET_U*2;
    for (int v=0; v<GRID_V; v++){
        for (int u=0; u<GRID_U; u++){
            gridOut[gridOutOffset + v*grid_size*2 + u*2] =      (*grids[0].hostPtr)[v][u][0].x;
            gridOut[gridOutOffset + v*grid_size*2 + u*2 +1] =   (*grids[0].hostPtr)[v][u][0].y;
        }
    }

    double stop = getTime();
    std::cout << "dev->host copy = " << finishedCopy.elapsedTime(startCopy) << std::endl << "total exec time = " << (stop - start) << std::endl;

    printGrid(*grids[0].hostPtr, "GPU - Cuda");
}

