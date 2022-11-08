#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include "cuda.h"
#include "cuda_runtime.h"

#define _HOST_DEVICE_METHOD_QUALIFIER __host__ __device__

// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)

#define cuda_err_chk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = false) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(1);
  }
}

#define PRINT_ERROR                                                    \
  do {                                                                 \
    fprintf(stderr, "Error at line %d, file %s (%d) [%s]\n", __LINE__, \
            __FILE__, errno, strerror(errno));                         \
    exit(1);                                                           \
  } while (0)

static std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}

/*Device function that returns how many SMs are there in the device/arch - it
 * can be more than the maximum readable SMs*/
__device__ __forceinline__ unsigned int getnsmid() {
  unsigned int r;
  asm("mov.u32 %0, %%nsmid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ unsigned int my_lanemask32_lt() {
  unsigned int lanemask32_lt;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
  return (lanemask32_lt);
}

/*Device function that returns the current SMID of for the block being run*/
__device__ __forceinline__ unsigned int getsmid() {
  unsigned int r;
  asm("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

/*Device function that returns the current warpid of for the block being run*/
__device__ __forceinline__ unsigned int getwarpid() {
  unsigned int r;
  asm("mov.u32 %0, %%warpid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ unsigned int getwarpsz() {
  unsigned int warpSize;
  asm volatile("mov.u32 %0, WARP_SZ;" : "=r"(warpSize));
  return warpSize;
}

/*Device function that returns the current laneid of for the warp in the block
 * being run*/
__device__ __forceinline__ unsigned int getlaneid() {
  unsigned int r;
  asm("mov.u32 %0, %%laneid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ int binary_search(int num_elements,
                                             const int *__restrict__ arr,
                                             int target) {
  int lo = 0, hi = num_elements;
  // find element in arr[i] where i in [lo, hi)
  // This below check covers all cases , so need to check
  // for mid=lo-(hi-lo)/2
  while (hi - lo > 1) {
    int mid = (hi + lo) / 2;
    if (arr[mid] <= target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// TODO: figure out metadata caching to optimize the performance
template <typename Idx, typename IdxPtr>
__device__ __forceinline__ Idx find_relational_compact_as_of_node_index(
    Idx idx_relation, Idx idx_node, const IdxPtr unique_srcs_and_dests_rel_ptr,
    const IdxPtr unique_srcs_and_dests_node_indices) {
  Idx idx_relation_offset = unique_srcs_and_dests_rel_ptr[idx_relation];
  Idx idx_relation_plus_one_offset =
      unique_srcs_and_dests_rel_ptr[idx_relation + 1];
  return idx_relation_offset +
         binary_search(idx_relation_plus_one_offset - idx_relation_offset,
                       &unique_srcs_and_dests_node_indices[idx_relation_offset],
                       idx_node);
}

// workaround to assert an if constexpr clause won't be emitted during compile
// time Use the following: static_assert(dependent_false<T>::value); See
// https://stackoverflow.com/a/53945555/5555077
template <class T>
struct dependent_false : std::false_type {};
#define CONSTEXPR_CLAUSE_NONREACHABLE(T, reason) \
  static_assert(dependent_false<T>::value && reason)
