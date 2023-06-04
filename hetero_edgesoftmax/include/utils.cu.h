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
#include "kernel_enums.h"

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

template <typename Idx, typename IdxPtr>
__device__ __forceinline__ Idx binary_search(Idx num_elements, const IdxPtr arr,
                                             Idx target) {
  Idx lo = 0, hi = num_elements;
  // find element in arr[i] where i in [lo, hi)
  // This below check covers all cases , so need to check
  // for mid=lo-(hi-lo)/2
  while (hi - lo > 1) {
    Idx mid = (hi + lo) / 2;
    if (arr[mid] <= target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename Idx, typename IdxPtr>
__device__ __forceinline__ Idx linear_search(Idx num_elements, const IdxPtr arr,
                                             Idx target, Idx resume_from) {
  for (Idx lo = resume_from; lo < num_elements; lo += 1) {
    if (arr[lo] > target) {
      return lo - 1;
    }
  }
}

// TODO: is there a way to map from (src idx, etype) instead of edge idx to (row
// index in the compact tensor)?
// TODO: optimize when warp coorperatively work on to reduce the last 4-5 global
// loads
// TODO: figure out metadata caching to optimize the performance
template <typename Idx, CompactAsOfNodeKind kind>
__device__ __forceinline__ Idx find_relational_compact_as_of_node_index(
    Idx idx_relation, Idx idx_node, Idx idx_edata,
    ETypeMapperData<Idx, kind> etype_mapper_data) {
  if constexpr (IsBinarySearch(kind)) {
    Idx idx_relation_offset =
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation];
    Idx idx_relation_plus_one_offset =
        etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation + 1];
    return idx_relation_offset +
           binary_search<Idx, Idx *>(
               idx_relation_plus_one_offset - idx_relation_offset,
               &(etype_mapper_data
                     .unique_srcs_and_dests_node_indices[idx_relation_offset]),
               idx_node);
  } else {
    return etype_mapper_data.edata_idx_to_inverse_idx[idx_edata];
  }
}

template <typename Idx>
__device__ __host__ __forceinline__ Idx ceil_div(const Idx a, const Idx b) {
  return (a + b - 1) / b;
}

template <typename Idx>
__device__ __host__ __forceinline__ constexpr Idx min2(const Idx a,
                                                       const Idx b) {
  return a < b ? a : b;
}

template <typename Idx>
__device__ __host__ __forceinline__ constexpr Idx max2(const Idx a,
                                                       const Idx b) {
  return a > b ? a : b;
}

// workaround to assert an if constexpr clause won't be emitted during compile
// time Use the following: static_assert(dependent_false<T>::value); See
// https://stackoverflow.com/a/53945555/5555077
// example https://godbolt.org/z/3cGdneEKM
#define CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(FLAG, reason)                   \
  static_assert(std::is_same<std::true_type,                               \
                             std::integral_constant<bool, FLAG>>::value && \
                reason)

#define CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(FLAG, reason)                    \
  static_assert(std::is_same<std::false_type,                              \
                             std::integral_constant<bool, FLAG>>::value && \
                reason)

#define CONSTEXPR_FALSE_CLAUSE_STATIC_ASSERT(FLAG, asserted_true_expression,  \
                                             reason)                          \
  static_assert(                                                              \
      std::is_same<std::true_type,                                            \
                   std::integral_constant<bool, FLAG>>::value /*unreachable*/ \
      || (/*or true*/ asserted_true_expression && reason))

#define CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(FLAG, asserted_true_expression,   \
                                            reason)                           \
  static_assert(                                                              \
      std::is_same<std::false_type,                                           \
                   std::integral_constant<bool, FLAG>>::value /*unreachable*/ \
      || (/*or true*/ asserted_true_expression && reason))

// can be used for both threadIdx and blockIdx
__device__ __forceinline__ uint
get_canonical_1D_threading_Idx(uint3 actual_threadIdx, uint3 actual_blockDim) {
  return actual_threadIdx.x + actual_threadIdx.y * actual_blockDim.x +
         actual_threadIdx.z * actual_blockDim.x * actual_blockDim.y;
}
// can be used for both threadIdx and blockIdx
__device__ __forceinline__ uint3 get_pretended_threading_Idx(
    uint3 actual_threadIdx, uint3 actual_blockDim, uint3 pretended_blockDim) {
  assert(pretended_blockDim.x * pretended_blockDim.y * pretended_blockDim.z ==
         actual_blockDim.x * actual_blockDim.y * actual_blockDim.z);
  uint canonical_1D_threadIdx =
      get_canonical_1D_threading_Idx(actual_threadIdx, actual_blockDim);
  uint3 pretended_threadIdx{
      .x = canonical_1D_threadIdx % pretended_blockDim.x,
      .y = (canonical_1D_threadIdx %
            (pretended_blockDim.x * pretended_blockDim.y)) /
           pretended_blockDim.x,
      .z = canonical_1D_threadIdx /
           (pretended_blockDim.x * pretended_blockDim.y)};
  return pretended_threadIdx;
}

// from
// https://github.com/gunrock/loops/blob/6169cf64d06e17b24b7a687fe0baf7ba2347002b/include/loops/schedule/group_mapped.hxx#L109
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

#if CUDART_VERSION >= 12000
namespace cg_x = cg;
#else
namespace cg_x = cooperative_groups::experimental;
#endif

// example and APIs more from
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-tile2
// #if CUDART_VERSION >= 12000
//     __shared__ cg_x::block_tile_memory<threads_per_block> shared_for_cg;
// #else
//     __shared__ cg_x::block_tile_memory<4/* by default 8*/, threads_per_block>
//     shared_for_cg;
// #endif
// cg::thread_block thb = cg_x::this_thread_block(shared_for_cg);
// auto tile = tiled_partition<128>(thb);
// unsigned long long meta_group_size() const: Returns the number of groups
// created when the parent group was partitioned. unsigned long long
// meta_group_rank() const: Linear rank of the group within the set of tiles
// partitioned from a parent group (bounded by meta_group_size) void sync()
// const: Synchronize the threads named in the group

// TODO: step 1: add thb as an argument to the sgemm exec_function and
// initialize it in every entry global function
// TODO: stpe 2: implement logic to figure out the occupancy and warn if it is
// reduced we may use the following function and compare whether there is any
// num_block difference when between  specialized kernels w/ and w/o the cg
// __shared__ usage
// __host__​cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock ( size_t*
// dynamicSmemSize, const void* func, int  numBlocks, int  blockSize) Returns
// dynamic shared memory available per block when launching numBlocks blocks on
// SM. error no == 1 if cannot arrange (cannot satisfy the requested block size,
// block num  for example) use cudaGetLastError to reset the error code or
// better using __host__​__device__​cudaError_t
// cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, const void*
// func, int  blockSize, size_t dynamicSMemSize )