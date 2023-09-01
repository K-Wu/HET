# from sputnik/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)

project(HETEROEDGESOFTMAX CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

#
# Build options.
#
option(BUILD_TEST "build sputnik test suite" ON)
option(BUILD_BENCHMARK "build sputnik benchmark suite" ON)
option(BUILD_GENERATED "build sputnik-based generated op code" ON)
option(BUILD_TORCH_API "build torch api library" ON)
option(RES_USAGE "show resource usage during nvcc compilation" OFF)
option(GITHUB_ACTION_TEST
       "turn on if the machine does not have cuda device, e.g., github action"
       OFF)

if(DEFINED CUDA_ARCHS)
  message(WARNING "CUDA_ARCHS is set as ${CUDA_ARCHS}.")
endif()

# from https://stackoverflow.com/a/69353718
if(GITHUB_ACTION_TEST)
  # set a fake cuda arch for github action test purpose
  set(CUDA_ARCHS 75)
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCHS}")
else()
  include(FindCUDA/select_compute_arch)
  cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  message("CUDA_ARCH_LIST = ${CUDA_ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
  # TODO: this scheme works on machine with single type of GPU. generalize it
  # set CUDA_ARCHS as a two-digit number for sputnik
  set(CUDA_ARCHS ${CUDA_ARCH_LIST})
endif()

# Pass in CUDA_ARCHS as a C++ macro
add_definitions(-DCUDA_ARCHS=${CUDA_ARCHS})

#
# Handle Dependencies.
#
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

#
# C++ and CUDA flags.
#

# Requires C++17.
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Wall -Wno-sign-compare -maxrregcount 128 --restrict --extended-lambda -std=c++17 -g -lineinfo"
)

# NB: add -O3 in release mode
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message("debug mode: -g -G enabled")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DENABLE_DEBUG_MACRO")
else()
  message("release mode: -O3 enabled")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

if(RES_USAGE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -res-usage")
endif()

#
# Project build.
#
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(hetero_edgesoftmax)
