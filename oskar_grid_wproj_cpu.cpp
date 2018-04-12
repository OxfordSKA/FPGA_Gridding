
/****************************************************************************************
 *
 * Created by the Numerical Algorithms Group Ltd, 2017.
 *
 ***************************************************************************************/

// Whether or not to use asserts to ensure array bounds aren't overrun etc.
// Some variables are only needed/defined if asserts are switched on.
//#define USEASSERTS

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#ifdef USEASSERTS
#include <assert.h>
#endif

#include "oskar_grid_wproj_cpu.hpp"


// How to sort Tiles by the number of visibilities they contain.
bool sortTilesByVis(Tile a, Tile b) { return a.vis > b.vis; }

// Find the region of the grid which will be updated by the visibilities we are processing.
void get_grid_hit_box(
      const int num_w_planes, 
      const int* support,
      const int num_vis,
      const float* uu, 
      const float* vv,
      const float* ww, 
      const double cell_size_rad,
      const double w_scale, 
      const int grid_size, 
      int * num_skipped,
      int &topx,
      int &topy,
      int &botx,
      int &boty
      )
{
  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;
      
  topx = 2*grid_size;
  topy = 2*grid_size;
  botx = -1;
  boty = -1;

  // OpenMP won't let us reduce on pointers/variables passed by reference
  // so we define some local variables to use instead.
  int ltopx=topx;
  int ltopy=topy;
  int lbotx=botx;
  int lboty=boty;
  int loc_num_skipped = 0; 

  // Loop over visibilities.
#pragma omp parallel for default(shared) reduction(+:loc_num_skipped)	\
  reduction(min:ltopx) reduction(min:ltopy)				\
  reduction(max:lbotx) reduction(max:lboty) schedule(guided,5)
  for (int i = 0; i < num_vis; i++)
    {
      float pos_u, pos_v, ww_i;
      int grid_u, grid_v, grid_w, wsupport;

      // Convert UV coordinates to grid coordinates.
      pos_u = -uu[i] * scale;
      pos_v = vv[i] * scale;
      ww_i = ww[i];

      grid_u = (int)round(pos_u) + g_centre;
      grid_v = (int)round(pos_v) + g_centre;
      grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); 
      if (grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      // Catch points that would lie outside the grid.
      wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
	  grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
	{
	  loc_num_skipped++;
	  continue;
	}


      int x = grid_u - wsupport;
      int y = grid_v - wsupport;
      ltopx = (x < ltopx ? x : ltopx);
      ltopy = (y < ltopy ? y : ltopy);

      x = grid_u + wsupport;
      y = grid_v + wsupport;
      lbotx = (x > lbotx ? x : lbotx);
      lboty = (y > lboty ? y : lboty);

    }

  // Update the result variables with the (reduced) local variables.
  *num_skipped = loc_num_skipped;
  topx=ltopx;
  topy=ltopy;
  botx=lbotx;
  boty=lboty;
}

// Count how many visibilities are in each Tile that the grid is divided into.
// Within these Tiles we'll also be sorting in order of wkernel so we'll be
// counting how many points are at each grid_w value within the Tile. This is
// known as a layered sort.
void oskar_count_elements_in_tiles_layered(
	const int num_w_planes, 
	const int* support,
	const int num_vis,
	const float* uu, 
	const float* vv,
	const float* ww, 
	const double cell_size_rad,
	const double w_scale, 
	const int grid_size, 
	const Point boxTop,
	const Point boxBot,
	const int tileWidth,
	const int tileHeight,
	const Point numTiles,
	std::vector<int> &numPointsInTilesLayered
	)
{
  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;

#define NUM_POINTS_IN_TILES_LAYERED(uu, vv, ww)  numPointsInTilesLayered.at( ( (uu) + (vv)*numTiles.u )*num_w_planes + ww)
#define NUM_POINTS_OUTSIDE_TILES_LAYERED(ww)     numPointsInTilesLayered.at( numTiles.v*numTiles.u*num_w_planes + ww )

  // Loop over visibilities.
#pragma omp parallel for default(shared) schedule(guided,5)
  for (int i = 0; i < num_vis; i++)
    {
      // Convert UV coordinates to grid coordinates.
      float pos_u = -uu[i] * scale;
      float pos_v = vv[i] * scale;
      float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale))); 
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      // Catch points that would lie outside the grid.
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
	  grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
	{
	  continue;
	}

      // We compute the following in floating point:
      //      boxTop.u + pu1*tileWidth <= grid_u - wsupport
      //                                   grid_u + wsupport <= boxTop.u + pu2*tileWidth - 1
      float fu1 = float(grid_u - wsupport - boxTop.u)/tileWidth;
      float fu2 = float(grid_u + wsupport - boxTop.u + 1)/tileWidth;
      // Intersect [fu1, fu2] with [0, numTiles.u)
      float fu_int[] = { (fu1<0.0f ? 0.0f: fu1), (fu2>numTiles.u ? numTiles.u : fu2) };
      int u_int[] = { floor(fu_int[0]), ceil(fu_int[1]) };

      float fv1 = float(grid_v - wsupport - boxTop.v)/tileHeight;
      float fv2 = float(grid_v + wsupport - boxTop.v + 1)/tileHeight;
      // Intersect [fv1, fv2] with [0, numTiles.v)
      float fv_int[] = { (fv1<0.0f ? 0.0f: fv1), (fv2>numTiles.v ? numTiles.v : fv2) };
      int v_int[] = { floor(fv_int[0]), ceil(fv_int[1]) };

      for (int pv=v_int[0]; pv < v_int[1]; pv++)
	{
	  for (int pu = u_int[0]; pu < u_int[1]; pu++)
	    {
              // Update the point counter.
#pragma omp atomic
	      NUM_POINTS_IN_TILES_LAYERED(pu, pv, grid_w) += 1;
	    }
	}
      // Now need to check whether this grid point would also have hit grid areas
      // not covered by any tiles.
      if(   grid_u-wsupport < boxTop.u ||
            grid_u+wsupport >= boxBot.u ||
            grid_v-wsupport < boxTop.v ||
            grid_v+wsupport >= boxBot.v )
	{
#pragma omp atomic
	  NUM_POINTS_OUTSIDE_TILES_LAYERED(grid_w)++;
	}
    }

#undef NUM_POINTS_IN_TILES_LAYERED
#undef NUM_POINTS_OUTSIDE_TILES_LAYERED
}

// Sort the visibilities in Tiles into buckets so that adjacent visibilities will be processed
// one after the other. Also sorts the visibilities within these buckets so that they are in order of wkernel level.
void oskar_bucket_sort_layered(
	   const int num_w_planes, 
	   const int* support,
	   const int num_vis,
	   const float* uu, 
	   const float* vv,
	   const float* ww, 
	   const float* vis, 
	   const float* weight, 
	   const double cell_size_rad,
	   const double w_scale, 
	   const int grid_size, 
	   const Point boxTop,
	   const Point boxBot,
	   const int tileWidth,
	   const int tileHeight,
	   const Point numTiles,
	   const std::vector<int> &offsetsPointsInTilesLayered,
	   std::vector<int> &wk_offsetsPointsInTilesLayered,
	   std::vector<float> & bucket_uu,
	   std::vector<float> & bucket_vv,
	   std::vector<float> & bucket_ww,
	   std::vector<float2> & bucket_vis,
	   std::vector<float> & bucket_weight
	   )
{
  auto start = std::chrono::high_resolution_clock::now();

  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;

#define WK_OFFSETS_IN_TILES_LAYERED(uu, vv, ww)  wk_offsetsPointsInTilesLayered.at( ((uu) + (vv)*numTiles.u)*num_w_planes + ww)
#define WK_OFFSETS_OUTSIDE_TILES_LAYERED(ww)     wk_offsetsPointsInTilesLayered.at( numTiles.u*numTiles.v*num_w_planes + ww )

  // Loop over visibilities.
#pragma omp parallel for default(shared) schedule(guided,5)
  for (int i = 0; i < num_vis; i++)
    {
      // Convert UV coordinates to grid coordinates.
      float pos_u = -uu[i] * scale;
      float pos_v = vv[i] * scale;
      float ww_i = ww[i];

      const int grid_u = (int)round(pos_u) + g_centre;
      const int grid_v = (int)round(pos_v) + g_centre;
      int grid_w = (int)round(sqrt(fabs(ww_i * w_scale)));
      if(grid_w >= num_w_planes) grid_w = num_w_planes - 1;

      // Catch points that would lie outside the grid.
      const int wsupport = support[grid_w];
      if (grid_u + wsupport >= grid_size || grid_u - wsupport < 0 ||
	  grid_v + wsupport >= grid_size || grid_v - wsupport < 0)
	{
	  continue;
	}

      float fu1 = float(grid_u - wsupport - boxTop.u)/tileWidth;
      float fu2 = float(grid_u + wsupport - boxTop.u + 1)/tileWidth;
      // Intersect [fu1, fu2] with [0, numTiles.u)
      float fu_int[] = { (fu1<0.0f ? 0.0f: fu1), (fu2>numTiles.u ? numTiles.u : fu2) };
      int u_int[] = { floor(fu_int[0]), ceil(fu_int[1]) };

      float fv1 = float(grid_v - wsupport - boxTop.v)/tileHeight;
      float fv2 = float(grid_v + wsupport - boxTop.v + 1)/tileHeight;
      // Intersect [fv1, fv2] with [0, numTiles.v)
      float fv_int[] = { (fv1<0.0f ? 0.0f: fv1), (fv2>numTiles.v ? numTiles.v : fv2) };
      int v_int[] = { floor(fv_int[0]), ceil(fv_int[1]) };


      for (int pv=v_int[0]; pv < v_int[1]; pv++)
	{
	  for (int pu = u_int[0]; pu < u_int[1]; pu++)
	    {
	      // Atomic: get current offset and increment offset by one.
	      int off;
#pragma omp atomic capture
	      off = WK_OFFSETS_IN_TILES_LAYERED(pu, pv, grid_w)++;
	   
#ifdef USEASSERTS
	      int idx = (pu + pv*numTiles.u)*num_w_planes + grid_w;
	      assert(off+1 <= offsetsPointsInTilesLayered.at(idx+1) );
#endif
	   
              // Add this visibility to the off^th position in the bucket arrays;
              // this ensures that the visibilities in the buckets arrays according to which Tile
              // they lie in and within that ordered by grid_w.
	      bucket_uu.at(off) = uu[i]; 
	      bucket_vv.at(off) = vv[i]; 
	      bucket_ww.at(off) = ww[i]; 

	      float2 v;
	      v.x = vis[2 * i];
	      v.y = vis[2 * i + 1];
	      bucket_vis.at(off) = v; 
	      bucket_weight.at(off) = weight[i];
	    }
	}
      
      // Now need to check whether this grid point would also have hit grid areas
      // not covered by any tiles.
      if(   grid_u-wsupport < boxTop.u ||
            grid_u+wsupport >= boxBot.u ||
            grid_v-wsupport < boxTop.v ||
            grid_v+wsupport >= boxBot.v )
	{
	  // Atomic: get current offset and increment offset by one.
	  int off;
#pragma omp atomic capture	
	  off = WK_OFFSETS_OUTSIDE_TILES_LAYERED(grid_w)++;
	
#ifdef USEASSERTS
	  int idx = numTiles.u*numTiles.v*num_w_planes + grid_w;
	  assert(off+1 <= offsetsPointsInTilesLayered.at(idx+1) );
	  assert(off   < bucket_uu.size() );
#endif
	 
	  bucket_uu.at(off) = uu[i]; 
	  bucket_vv.at(off) = vv[i]; 
	  bucket_ww.at(off) = ww[i]; 

	  float2 v;
	  v.x = vis[2 * i];
	  v.y = vis[2 * i + 1];
	  bucket_vis.at(off) = v; 
	  bucket_weight.at(off) = weight[i];
	}
    }

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop-start;
  printf("Bucket sort time: %gms\n", diff.count()*1000);

#undef WK_OFFSETS_IN_TILES_LAYERED
#undef WK_OFFSETS_OUTSIDE_TILES_LAYERED  
}

// Look up which Tile a particular visibility is located in .
int tileLookup(
        const std::vector<int> tileOff,
        const int visibility
        )
{
  //First entry in tileOff is always 0.
  for(int tile=0; tile<tileOff.size(); tile++){
    if(tileOff.at(tile)>visibility)
      return tile-1;
  }
  return tileOff.size()-1;
}

// Process the grid Tiles 
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
	   int numTiles_u, int numTiles_v,
	   const int* numPointsInTiles,
	   const int* offsetsPointsInTiles,
	   const float* bucket_uu,
	   const float* bucket_vv,
	   const float* bucket_ww,
	   const float2* bucket_vis,
	   const float* bucket_weight,
       const int *workQueue_pu, const int *workQueue_pv,
	   double* norm, 
	   float* grid
	   )
{

  const int g_centre = grid_size / 2;
  const double scale = grid_size * cell_size_rad;

  const int nTiles = numTiles_u * numTiles_v;

  double local_norm = 0.0;
  
#define NUM_POINTS_IN_TILES(uu, vv)  numPointsInTiles[( (uu) + (vv)*numTiles_u)]
#define OFFSETS_IN_TILES(uu, vv)     offsetsPointsInTiles[( (uu) + (vv)*numTiles_u)]

  auto start = std::chrono::high_resolution_clock::now();

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

	  const int kstart = std::max(tile_u[0]-grid_u, -wsupport);
	  const int kend   = std::min(tile_u[1]-grid_u,  wsupport);
    
	  const int jstart = std::max(tile_v[0]-grid_v, -wsupport);
	  const int jend   = std::min(tile_v[1]-grid_v,  wsupport);

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
	  local_norm += sum * w;
	}
    }

  *norm += local_norm;

  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = stop-start;
  printf("Tile processing time: %gms\n", diff.count()*1000);
   
#undef NUM_POINTS_IN_TILES
#undef OFFSETS_IN_TILES
}

// Compact the wkernels so that only the values for the wsupport of each kernel level are stored,
// rather than storing all kernels with max(wsupport) elements. Also reorder and copy the elements of
// each kernel so that they are stored in a more cache- and vectorisation-friendly manner.
void compact_oversample_wkernels(
	 const int num_w_planes, 
	 const int* wkernel_support,
	 const int oversample, 
	 const int conv_size_half,
	 const float* conv_func, 
	 std::vector<float2> &compacted_oversampled_wkernels,
	 std::vector<int>   &compacted_oversampled_wkernel_start_idx
	 )
{
  const int kernel_dim = conv_size_half * conv_size_half;

  std::vector<int> compacted_wkernel_start_idx;
  std::vector<float2> compacted_wkernels;

  // Each slice of wkernel data occupies conv_size_half^2 elements in memory.
#define WKERNEL(kidx,iy,ix,r)  conv_func[2*( (kidx)*kernel_dim + (iy)*conv_size_half + (ix) ) + (r)]

  // Inside each kernel, we only access elements at locations
  //
  //    id = abs(off + i*oversample)
  //
  // where 
  //
  //    off \in [-oversample/2, oversample/2] 
  //    
  // and
  //
  //    j = -wsupport, ..., -1, 0, 1, ..., wsupport
  //
  // This means we access locations between
  //
  //   id = 0  and  id = oversample/2 + wsupport*oversample
  //
  // Since we do this in u and v dimensions the size of each 
  // compacted wkernel is
  // 
  //        (oversample/2 + wsupport*oversample + 1)^2
  //
  // float2 values, since we are dealing with complex data

  compacted_wkernel_start_idx.resize(num_w_planes);
  compacted_oversampled_wkernel_start_idx.resize(num_w_planes);

  int compacted_size = 0;
  int compacted_oversampled_size = 0;
  
  for(int grid_w = 0; grid_w < num_w_planes; grid_w++)
    {
      int wsupport = wkernel_support[grid_w];
      int size = oversample/2 + wsupport*oversample + 1;
      size = size * size;

      compacted_wkernel_start_idx.at(grid_w) = compacted_size;
      compacted_size += size;
      compacted_oversampled_wkernel_start_idx.at(grid_w) = compacted_oversampled_size;
      compacted_oversampled_size += (2*wsupport+1) * (2*wsupport + 1) * (oversample/2 + 1) * (oversample/2 + 1);
    }

  // Allocate memory.
  compacted_wkernels.resize(compacted_size);
  compacted_oversampled_wkernels.resize(compacted_oversampled_size);

  // Loop through the kernel and store the values in the new compacted kernel array.
#pragma omp parallel default(shared)
  {
#pragma omp for schedule(static)
    for(int grid_w = 0; grid_w < num_w_planes; grid_w++) 
      {
	const int wsupport = wkernel_support[grid_w];
	const int start = compacted_wkernel_start_idx.at(grid_w);
	const int size = oversample/2 + wsupport*oversample + 1;

	for(int iy = 0; iy < size; iy++) {
	  for(int ix=0; ix < size; ix++) {
            float2 c;
            c.x = WKERNEL(grid_w,iy,ix,0);
            c.y = WKERNEL(grid_w,iy,ix,1);

            compacted_wkernels.at(start + iy*size + ix) = c;
#ifdef USEASSERTS
            if(grid_w < num_w_planes-1) assert(start + iy*size + ix < compacted_wkernel_start_idx.at(grid_w+1));
#endif
	  }
	}
      }

    // The compacted kernel is very space-efficient, but not cache-efficient as it won't be accessed with regular strides
    // Loop through the compacted wkernel in the order in which its elements would actually be accessed and store this in
    // compacted_oversampled_wkernel. This stores some of the elements twice but is more cache-friendly.
#pragma omp for schedule(static)
    for (int grid_w = 0 ; grid_w < num_w_planes ; grid_w++)
      {
	int wsupport = wkernel_support[grid_w];
	int compact_start = compacted_wkernel_start_idx.at(grid_w);
	int compact_oversample_start = compacted_oversampled_wkernel_start_idx.at(grid_w);
	int size = oversample/2 + oversample*wsupport + 1;
       
        // Store a version for each possible offset in the u and v directions.
	for (int off_v = 0 ; off_v <= oversample/2 ; off_v++)
	  {
	    for (int off_u = 0 ; off_u <= oversample/2 ; off_u++)
	      {
                // Now loop through the support of the kernel in the u and v directions.
		for (int j = -wsupport ; j <= wsupport ; j++)
		  {
		    int iy = abs(off_v + j * oversample);
		    for (int i = -wsupport ; i <= wsupport ; i++)
		      {
			int ix = abs(off_u + i * oversample);
			int p = compact_start + iy*size + ix;
			int q = compact_oversample_start + (off_v)*(oversample/2 + 1)*(2*wsupport + 1)*(2*wsupport + 1)  + (off_u)*(2*wsupport + 1)*(2*wsupport + 1)  + (j + wsupport)*(2*wsupport + 1) + i + wsupport;
			compacted_oversampled_wkernels.at(q) = compacted_wkernels.at(p);
		      }
		  }
	      }
	  }
      }
  } //end parallel

  //printf("Compacted Wkernels:\n\toriginal wkernel stack = %gMB\n\tcompacted wkernel stack = %gMB\n\tcompacted oversampled wkernel stack = %gMB\n", 
  //       2.0*kernel_dim*num_w_planes*4.0/1024.0/1024.0, compacted_wkernels.size()*8.0/1024.0/1024.0, compacted_oversampled_wkernels.size()*8.0/1024.0/1024.0);

#undef WKERNEL
}

// The main function that does the gridding by calling all the functions defined above.
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
	size_t* global_num_skipped,
	double* norm, 
	float* grid
	)
{

  std::vector<int> compacted_wkernel_start;
  std::vector<float2> compacted_wkernels;

  double time1, time2;
  double time3, time4;
   
  //time1 = omp_get_wtime();

  // First of all, reduce the amount of space that the wkernels occupy and reorder them
  // to be more cache and vectorisation friendly. As this could be done at the point the
  // kernels are tabulated, we exclude the time taken for this function from our overall timing results.
  compact_oversample_wkernels(num_w_planes, support, oversample, conv_size_half, conv_func, compacted_wkernels, compacted_wkernel_start);

  //time2 = omp_get_wtime() - time1;
  //printf("Compacting time: %fms\n",time2*1000);

  //time1 = omp_get_wtime();
  
  *norm = 0.0;

  int max_wsupport = 0;
  for(int i=0; i<num_w_planes; i++) {
    max_wsupport = std::max(max_wsupport, support[i]);
  }

  // Get the bounding box for the area of the grid that actually gets updated.
  Point topLeft(2*grid_size, 2*grid_size);
  Point botRight(-1, -1);

  //time3 = omp_get_wtime();
   
  int num_skipped_val=0;
  int *num_skipped = &num_skipped_val;
  get_grid_hit_box(num_w_planes, support, num_vis, uu, vv, ww, cell_size_rad, w_scale, grid_size, 
		   num_skipped, topLeft.u, topLeft.v, botRight.u, botRight.v);
  
  *global_num_skipped = *num_skipped;

  //time4 = omp_get_wtime() - time3;
  //printf("Finding grid hit box time: %fms\n",time4*1000);

  const float factor = 0.0f;
  int len_u = int( factor*(botRight.u - topLeft.u) );
  int len_v = int( factor*(botRight.v - topLeft.v) );
  Point boxTop( topLeft.u + len_u, topLeft.v + len_v );
  Point boxBot( botRight.u - len_u, botRight.v - len_v );
     
  // We now fix our Tile size. This is a tunable parameter.
  const int tileWidth = 64;
  const int tileHeight = 64;
   
  // Work out the number of tiles we'll have in the central box.
  const Point numTiles( (boxBot.u-boxTop.u + tileWidth-1)/tileWidth, (boxBot.v-boxTop.v + tileHeight-1)/tileHeight );

  // Adjust our box to be a whole number of Tiles.
  boxBot.u = boxTop.u + numTiles.u*tileWidth;
  boxBot.v = boxTop.v + numTiles.v*tileHeight;
   
  // We now need to count how many points are in each Tile so that we can sort them.
  const int nTiles = numTiles.u * numTiles.v;
  std::vector<int> numPointsInTiles(nTiles + 1);
  std::vector<int> numPointsInTilesLayered((nTiles + 1)*num_w_planes);
  numPointsInTiles.resize(nTiles+1, 0);
  numPointsInTilesLayered.resize((nTiles+1)*num_w_planes, 0);

  //time3 = omp_get_wtime();

  // Count how many visibilities fall into each Tile
  oskar_count_elements_in_tiles_layered(num_w_planes, support, num_vis, uu, vv, ww, cell_size_rad, w_scale, 
			     grid_size, boxTop, boxBot, tileWidth, tileHeight, numTiles, numPointsInTilesLayered);

  //time4 = omp_get_wtime() - time3;
  //printf("Counting elements in tiles in: %fms\n",time4*1000);
   
  // Store the (pu,pv,num_vis) for each Tile, in order to form a work queue later that will
  // tell us which order to process the Tiles in to try to ensure load balance.
  // Note we don't want to put nPIT(numTiles) in here as it stores the #vis outside all Tiles
  // and we're not going to process those (it should be 0)
  std::vector<Tile> workQueue(nTiles);
   
  // Prepare for the bucket sort
  std::vector<int> offsetsPointsInTiles( nTiles+2 );
  std::vector<int> offsetsPointsInTilesLayered( (nTiles+2)*num_w_planes );
  int totalVisibilities = 0;
  int sum = 0;
  for(int i=0; i<numPointsInTiles.size(); i++) {
    sum = 0;
    for (int w = 0 ; w < num_w_planes ; w++) {
      int idx = i*num_w_planes + w;
      offsetsPointsInTilesLayered.at(idx) = totalVisibilities + sum;
      sum += numPointsInTilesLayered.at(idx);
    }
    numPointsInTiles.at(i) = sum;
    // Note that after this loop totalVisibilities >= num_vis as if a visibility affects more than one Tile (owing to the
    // support of the wkernel) then it is counted more than once.
    offsetsPointsInTiles.at(i) = totalVisibilities;
    totalVisibilities += sum;

    // Populate the work queue with info about this Tile.
    // Exclude final Tile as this is "everything outside the grid"
    // and we don't want to do any work on those points.
    if(i<nTiles){
      workQueue.at(i).pu = i%numTiles.u;
      workQueue.at(i).pv = i/numTiles.u;
      workQueue.at(i).vis = sum;
    }
  }

  /*
  // A useful loop for printing information about the Tiles and workQueue if needed.
  for(int pv=0; pv<numTiles.v; pv++) {
    for(int pu=0; pu<numTiles.u; pu++) {
      printf("\tTile count(%d,%d) \t\t= %d\n",pv,pu,numPointsInTiles.at(pu + pv*numTiles.u));
      printf("\tWork queue(%d,%d) \t\t= %d\n",workQueue.at(pu + pv*numTiles.u).pv,workQueue.at(pu + pv*numTiles.u).pu,workQueue.at(pu + pv*numTiles.u).vis);
    }
  }
  */

  // Sort the work queue into descending order by number of visibilities.
  // This will hopefully make a sensible order in which to process them.
  std::sort(workQueue.begin(), workQueue.end(), sortTilesByVis);
  // put workQueue into format that can be sent to an OpenCl kernel
  std::vector<int> workQueue_pu(nTiles);
  std::vector<int> workQueue_pv(nTiles);
  for (int tileIndex=0; tileIndex<nTiles; tileIndex++){
      workQueue_pu[tileIndex] = workQueue[tileIndex].pu;
      workQueue_pv[tileIndex] = workQueue[tileIndex].pv;
  }

   
#ifdef USEASSERTS
  assert(offsetsPointsInTiles.at(nTiles+1)==0);
#endif

  offsetsPointsInTiles.at(nTiles+1) = totalVisibilities;
  // Create some arrays for the bucket sort.
  std::vector<float> bucket_uu(totalVisibilities), bucket_vv(totalVisibilities), bucket_ww(totalVisibilities), bucket_weight(totalVisibilities);
  std::vector<float2> bucket_vis(totalVisibilities);
  std::vector<int> wk_offsetsPointsInTiles( offsetsPointsInTiles.size() );
  std::vector<int> wk_offsetsPointsInTilesLayered( offsetsPointsInTilesLayered.size() );
  for(int i=0; i<offsetsPointsInTiles.size(); i++) wk_offsetsPointsInTiles.at(i) = offsetsPointsInTiles.at(i);
  for(int i=0; i<offsetsPointsInTilesLayered.size(); i++) wk_offsetsPointsInTilesLayered.at(i) = offsetsPointsInTilesLayered.at(i);

  // Do the bucket sort. This is a layered sort: we return the bucket arrays with the visibilities sorted according
  // to the Tiles affected and, within each Tile, also in order of grid_w, the layer of the wkernel used.
  oskar_bucket_sort_layered(num_w_planes, support, num_vis, uu, vv, ww, vis, weight, cell_size_rad,
				w_scale, grid_size, 
				boxTop, boxBot, tileWidth, tileHeight, numTiles, 
				offsetsPointsInTilesLayered, wk_offsetsPointsInTilesLayered,
			        bucket_uu, bucket_vv, bucket_ww, bucket_vis, bucket_weight);
      
  // Check we actually have some work to do.
  // If we've set factor = 1 then there won't be any....
  if(numTiles.u > 0 && numTiles.v > 0) 
    {
      // Process the whole grid
      oskar_process_all_tiles(num_w_planes, support, oversample, compacted_wkernel_start.data(),
						   compacted_wkernels.data(), cell_size_rad, w_scale, grid_size,
						   boxTop.u, boxTop.v, tileWidth, tileHeight, numTiles.u, numTiles.v,
						   numPointsInTiles.data(), offsetsPointsInTiles.data(), 
						   bucket_uu.data(), bucket_vv.data(), bucket_ww.data(), bucket_vis.data(), 
                           bucket_weight.data(),
						   workQueue_pu.data(), workQueue_pv.data(), norm, grid);
      
    }

  //time2 = omp_get_wtime() - time1;

  //printf("Total time excluding compacting: %f seconds\n", time2);


  return;
}



