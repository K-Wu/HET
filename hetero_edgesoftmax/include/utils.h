#pragma once
#include <mutex>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <tuple>
#include "cuda.h"

#define _HOST_DEVICE_METHOD_QUALIFIER __host__ __device__

#define cuda_err_chk(ans)                     \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{

    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(1);
    }
}

#define PRINT_ERROR                                              \
    do                                                           \
    {                                                            \
        fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", \
                __LINE__, __FILE__, errno, strerror(errno));     \
        exit(1);                                                 \
    } while (0)

static std::chrono::time_point<std::chrono::high_resolution_clock> now()
{
    return std::chrono::high_resolution_clock::now();
}

/*Device function that returns how many SMs are there in the device/arch - it can be more than the maximum readable SMs*/
__device__ __forceinline__ unsigned int getnsmid()
{
    unsigned int r;
    asm("mov.u32 %0, %%nsmid;"
        : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned int my_lanemask32_lt()
{
    unsigned int lanemask32_lt;
    asm volatile("mov.u32 %0, %%lanemask_lt;"
                 : "=r"(lanemask32_lt));
    return (lanemask32_lt);
}

/*Device function that returns the current SMID of for the block being run*/
__device__ __forceinline__ unsigned int getsmid()
{
    unsigned int r;
    asm("mov.u32 %0, %%smid;"
        : "=r"(r));
    return r;
}

/*Device function that returns the current warpid of for the block being run*/
__device__ __forceinline__ unsigned int getwarpid()
{
    unsigned int r;
    asm("mov.u32 %0, %%warpid;"
        : "=r"(r));
    return r;
}

__device__ __forceinline__ unsigned int getwarpsz()
{
    unsigned int warpSize;
    asm volatile("mov.u32 %0, WARP_SZ;"
                 : "=r"(warpSize));
    return warpSize;
}

/*Device function that returns the current laneid of for the warp in the block being run*/
__device__ __forceinline__ unsigned int getlaneid()
{
    unsigned int r;
    asm("mov.u32 %0, %%laneid;"
        : "=r"(r));
    return r;
}
