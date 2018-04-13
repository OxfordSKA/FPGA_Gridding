#! CMake initialisation.
cmake_minimum_required(VERSION 3.1)
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 OLD)
endif()

#! Declare the CMake project.
project(grid_visibilities C CXX)
set (CMAKE_CXX_STANDARD 11)

#! Find Threads.
find_package(Threads REQUIRED)

#! Find OpenCL
#find_package(OpenCL REQUIRED)

set(AOCL_COMPILE_CONFIG "-I/usr/local/intelFPGA_pro/17.1/hld/host/include")
set(AOCL_LINK_CONFIG "-L/usr/local/intelFPGA_pro/17.1/hld/board/a10_ref/linux64/lib -L/usr/local/intelFPGA_pro/17.1/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -laltera_a10_ref_mmd -lelf")

#! Compiler flags.
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*"
            OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # Enable all warnings.
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC ${AOCL_COMPILE_CONFIG} ${AOCL_LINK_CONFIG}")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#! Build type detection.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
    if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

#! Build cfitsio library.
add_subdirectory(cfitsio-3.41)

#! Build oskar_binary library.
add_subdirectory(utils/binary)

#! Build test visibility gridder.
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/cfitsio-3.41)
include_directories(${PROJECT_SOURCE_DIR}/AOCL_UTILS)
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIR})

message(status " include dir: " ${OpenCL_INCLUDE_DIR})

set(read_vis_SRC
    # List of source files to compile.
    utils/check_value.c
    utils/oskar_thread.c
    utils/oskar_timer.c
    utils/oskar_grid_weights.c
    utils/read_kernel.c
    utils/read_vis.c
    utils/write_fits_cube.c
    oskar_grid_wproj.c
    oskar_grid_wproj_fpga.cpp
    AOCL_UTILS/opencl.cpp
    AOCL_UTILS/options.cpp
    main.cpp
)
add_executable(main ${read_vis_SRC})

target_link_libraries(main cfitsio oskar_binary m Threads::Threads)
