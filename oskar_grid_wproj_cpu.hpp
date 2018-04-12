
/****************************************************************************************
 *
 * Created by the Numerical Algorithms Group Ltd, 2017.
 *
 ***************************************************************************************/

#pragma once

#include <vector>

// We are dealing with Complex numbers, so define a struct that associates
// 2 floats together (as an alternative to a Complex class).
struct float2 {
  float x;
  float y;
};

// A point on the grid with coordinates u and v.
struct Point {
  int u;
  int v;
  Point(int u, int v) : u(u), v(v) {}
  Point(const Point& copy) : u(copy.u), v(copy.v) { }
};

// A rectangular region of the grid defined in terms of its top-left and bottom-right points.
struct Box {
   Point topLeft;
   Point botRight;
   Box(const Point &topLeft, const Point &bottomRight) : topLeft(topLeft), botRight(bottomRight) {  }
   Box(const Point &center, int wsupport) : topLeft(center.u-wsupport, center.v-wsupport), 
   botRight(center.u+wsupport,center.v+wsupport) {}
};

// Visibilities near each other are handled in batches called Tiles.
// Each Tile has a location in the grid of Tiles recorded as pu and pv
// The number of visibilities contained in a Tile is recorded in vis.
struct Tile {
  int pu;
  int pv;
  int vis;
};

// How to sort Tiles by the number of visibilities they contain.
bool sortTilesByVis(
     Tile a,
     Tile b
     );

// Look up which Tile a particular visibility is located in.
int tileLookup(
      const std::vector<int> tileOff,
      const int visibility
      );

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
      );

// Count how many visibilities are in each Tile that the grid is divided into.
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
      );

// Sort the visibilities in Tiles into buckets so that adjacent visibilities will be processed
// one after the other. Also sorts the visibilities so that within each Tile they are in order
// of wkernel level.
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
      );

// Process the Tiles outside of the central box.
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
       const float* bucket_weight,
       const int *workQueue_pu, const int *workQueue_pv,
       double* norm,
       float* grid
       );

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

