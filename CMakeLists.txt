# from sputnik/CMakeLists.txt
cmake_minimum_required(VERSION 3.13)

project(HETEROEDGESOFTMAX CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

##
### Build options.
##
# option(BUILD_TEST "Build test suite" OFF)
option(BUILD_TEST "build sputnik test suite" ON)
option(BUILD_BENCHMARK "build sputnik benchmark suite" ON)
option(BUILD_GENERATED "build sputnik-based generated op code" ON)
option(BUILD_TORCH_API "build torch api library" ON)
option(RES_USAGE "show resource usage during nvcc compilation" OFF)

##
### Handle Dependencies.
##

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

##
### C++ and CUDA flags.
##

# Requires C++17.
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wno-sign-compare -maxrregcount 128 -g -G -lineinfo --restrict --extended-lambda -std=c++17")
#TODO: add  -O3 in release mode 

if (RES_USAGE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -res-usage")
endif()

#if(BUILD_TEST)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPUTNIK_BUILD_TEST")
#  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSPUTNIK_BUILD_TEST")
#endif()


#if(NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 17)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif()

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

##
### Project build.
##
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(hetero_edgesoftmax)