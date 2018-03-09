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

#if 1

#define REAL float
#define REAL2 float2
#define REAL3 float3
#define REAL4 float4
#define MAKE_REAL2(A,B) make_float2(A,B)
#define MAKE_REAL3(A,B,C) make_float3(A,B,C)

#else

#define REAL double
#define REAL2 double2
#define REAL3 double3
#define REAL4 double4
#define MAKE_REAL2(A,B) make_double2(A,B)
#define MAKE_REAL3(A,B,C) make_double3(A,B,C)

#endif


#define FAKE_VIS_VALUES 0
#define FAKE_KERNEL_VALUES 0

#define MODE_SIMPLE	 0
#define MODE_OVERSAMPLE	 1
#define MODE_INTERPOLATE 2

#if !defined MODE
#define MODE		MODE_OVERSAMPLE
#endif

#undef DEGRIDDING
#define USE_REAL_UVW
#define USE_SYMMETRY
#undef MAP_OBJECTS
#define ENABLE_PROFILING

#if defined __CUDA__ || 0
#define USE_TEXTURE
#endif

#if !defined GRID_U
#define GRID_U		5000
#endif

#if !defined GRID_V
#define GRID_V		5500
#endif

#if !defined TRIMMED_REGION_OFFSET_U
//#define TRIMMED_REGION_OFFSET_U		7564    // 56-82
#define TRIMMED_REGION_OFFSET_U		7064        // 56-82

#endif

#if !defined TRIMMED_REGION_OFFSET_V
//#define TRIMMED_REGION_OFFSET_V		6759    // 56-82
#define TRIMMED_REGION_OFFSET_V		6059        // 56-82
#endif

#define POLARIZATIONS	1

#if !defined SUPPORT_U
#define SUPPORT_U	146
#endif
#if !defined X_SUPPORT
#define X_SUPPORT SUPPORT_U
#endif

#if !defined SUPPORT_V
#define SUPPORT_V	146
#endif

#if !defined W_PLANES
#define W_PLANES	1201
#endif

#define OVERSAMPLE_U	4
#define OVERSAMPLE_V	4

#define CELL_SIZE_U	(1.08*13107.2 / GRID_U)
#define CELL_SIZE_V	(1.08*13107.2 / GRID_V)
#define CELL_SIZE_W	(8192.0 / W_PLANES)

#define NR_STATIONS	512
#define BASELINES	(NR_STATIONS * (NR_STATIONS - 1) / 2)
#define MAX_BASELINE_LENGTH	22000

#if !defined CHANNELS
#define CHANNELS	1
#endif

#if !defined TIMESTEPS
#define TIMESTEPS	1
#endif

#if !defined BLOCKS
#define BLOCKS		240
#endif

#define	STREAMS		1

#define SPEED_OF_LIGHT	299792458.

#if !defined ORDER
#define ORDER			ORDER_W_OV_OU_V_U
#endif

#define ORDER_W_OV_OU_V_U	0
#define ORDER_W_V_OV_U_OU	1
