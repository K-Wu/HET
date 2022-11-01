#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "utils.cu.h"

template <bool scatter_col_flag, int OUT_DIM>
__device__ __forceinline__ static float &GetCEle(float *C,
                                                 int *col_scatter_list, int K,
                                                 int idx_head, int row,
                                                 int col) {
  // We need to return an element reference so that we can store the result in
  // it.
  if constexpr (scatter_col_flag) {
    return C[(idx_head * K) + (row) + (col_scatter_list[col]) * OUT_DIM];
  } else {
    // #define C(idx_head, row, col) C[(idx_head * k) + (row) + (col)*OUT_DIM]
    return C[(idx_head * K) + (row) + (col)*OUT_DIM];
  }
}

template <bool gather_col_flag, int OUT_DIM>
__device__ __forceinline__ static float _basic_GetBEle(float *B,
                                                       int *col_gather_list,
                                                       int K, int idx_head,
                                                       int row, int col) {
  if constexpr (gather_col_flag) {
    return B[(idx_head * K) + (row) + (col_gather_list[col]) * OUT_DIM];
  } else {
    return B[(idx_head * K) + (row) + (col)*OUT_DIM];
  }
}

template <bool gather_col_flag, int OUT_DIM,
          bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
__device__ __forceinline__ static float GetBEle(
    float *B, int *col_gather_list, int *second_col_gather_list,
    int second_col_gather_list_length, int K, int idx_head, int row, int col) {
  if constexpr (gather_col_flag) {
    if constexpr (B_col_second_indirection_gather_flag) {
      if constexpr (B_col_second_indirection_gather_binary_search_flag) {
        int col_idx = col_gather_list[col];
        int col_idx_second = second_col_gather_list[col_idx];
        return B[(idx_head * K) + (row) + (col_idx_second)*OUT_DIM];
      } else {
        int col_idx = col_gather_list[col];
        int col_idx_second = binary_search(second_col_gather_list_length,
                                           second_col_gather_list, col_idx);
        return B[(idx_head * K) + (row) + (col_idx_second)*OUT_DIM];
      }
    } else {
      return B[(idx_head * K) + (row) + (col_gather_list[col]) * OUT_DIM];
    }
  } else {
    return B[(idx_head * K) + (row) + (col)*OUT_DIM];
  }
}

template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS,
          bool B_col_gather_flag, bool C_col_scatter_flag,
          bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    assert(0 && "not implemented");
    // static_assert(0, "not implemented");
  }
};

// when num_head==1, the function is reduced to a general matrix-matrix
// multiplication kernel from
// https://github.com/K-Wu/gpu-algorithms-labs/blob/master/labs/sgemm-regtiled-coarsened/template.cu
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, 1, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use register and shared memory tiling and thread coarsening
     *
     * NOTE: A and C are column major, B is column major as well
     *
     ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col)*m]
    constexpr int TILE_SZ_RATIO = (TILE_SZ_A / TILE_SZ_B);
    __shared__ float shmem[TILE_SZ_RATIO][TILE_SZ_B];
    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.x * TILE_SZ_A + threadIdx.x;

    for (int i = 0; i < (k + TILE_SZ_RATIO - 1) / TILE_SZ_RATIO; i++) {
      // load A in registers
      float reg0 = 0.0f;
      float reg1 = 0.0f;
      float reg2 = 0.0f;
      float reg3 = 0.0f;
      float reg4 = 0.0f;
      float reg5 = 0.0f;
      float reg6 = 0.0f;
      float reg7 = 0.0f;
      if (ArowIdx < m) {
        reg0 = (k > i * TILE_SZ_RATIO) ? A(ArowIdx, i * TILE_SZ_RATIO) : 0.0f;
        reg1 = (k > i * TILE_SZ_RATIO + 1) ? A(ArowIdx, i * TILE_SZ_RATIO + 1)
                                           : 0.0f;
        reg2 = (k > i * TILE_SZ_RATIO + 2) ? A(ArowIdx, i * TILE_SZ_RATIO + 2)
                                           : 0.0f;
        reg3 = (k > i * TILE_SZ_RATIO + 3) ? A(ArowIdx, i * TILE_SZ_RATIO + 3)
                                           : 0.0f;
        reg4 = (k > i * TILE_SZ_RATIO + 4) ? A(ArowIdx, i * TILE_SZ_RATIO + 4)
                                           : 0.0f;
        reg5 = (k > i * TILE_SZ_RATIO + 5) ? A(ArowIdx, i * TILE_SZ_RATIO + 5)
                                           : 0.0f;
        reg6 = (k > i * TILE_SZ_RATIO + 6) ? A(ArowIdx, i * TILE_SZ_RATIO + 6)
                                           : 0.0f;
        reg7 = (k > i * TILE_SZ_RATIO + 7) ? A(ArowIdx, i * TILE_SZ_RATIO + 7)
                                           : 0.0f;
      }
      // load B in shared memory
      int shdmemLDBrowIdx = i * TILE_SZ_RATIO + threadIdx.x / TILE_SZ_B;
      int shdmemLDBcolIdx = blockIdx.y * TILE_SZ_B + threadIdx.x % TILE_SZ_B;
      shmem[threadIdx.x / TILE_SZ_B][threadIdx.x % TILE_SZ_B] =
          (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, 0, shdmemLDBrowIdx,
                    shdmemLDBcolIdx)
              : 0.0f;

      __syncthreads();
      // compute C
      if (ArowIdx < m) {
        for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++) {
          int CcolIdx = shdmemColIdx + blockIdx.y * TILE_SZ_B;
          if (CcolIdx < n) {
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg0 * shmem[0][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg1 * shmem[1][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg2 * shmem[2][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg3 * shmem[3][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg4 * shmem[4][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg5 * shmem[5][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg6 * shmem[6][shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(C, C_col_scatter_list, k, 0,
                                                 ArowIdx, CcolIdx) +=
                reg7 * shmem[7][shdmemColIdx];
          }
        }
      }
      __syncthreads();
    }
  }
#undef A
};

template <int OUT_DIM, int NUM_HEADS, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<512, 32, OUT_DIM, NUM_HEADS, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    constexpr int TILE_SZ_A = 512;
    constexpr int TILE_SZ_B = 32;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int TILE_NUM_HEAD = 4;
    constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
    constexpr int K = 64;
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    assert(k == 64);
    assert(m == OUT_DIM);
/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1,
// ... -| |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest
// head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src head 1
// dest head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src
// head 1 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest head 0,
// src head 2 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest
// head 0, src head 2 dest head 1, ...  | |  src head 3 dest head 0, src head 3
// dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1,
// ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node
// 2 head 0, ... -| |  intermediate node 0 head 0, intermediate node 1 head 0,
// intermediate node 2 head 0, ...  | |  intermediate node 0 head 1,
// intermediate node 1 head 1, intermediate node 2 head 1, ...  | | intermediate
// node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node
// 2 head 2, ...  | |  intermediate node 0 head 2, intermediate node 1 head 2,
// intermediate node 2 head 2, ...  | |  intermediate node 0 head 3,
// intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node
// 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * K) + (row) + (col)*OUT_DIM]

    __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B][8];
    __shared__ float shmem_output[16 /*node idx*/]
                                 [16 /*element idx in 4 heads*/]
                                 [2 /*node idx 2nd part*/]
                                 [16 /*element idx in 4 heads 2nd part*/];
    for (int idx = 0; idx < 16; idx++) {
      shmem_output[idx][threadIdx.x / 32][threadIdx.x % 32 / 16]
                  [threadIdx.x % 16] = 0.0f;
    }
    static_assert(TILE_SZ_RATIO / TILE_NUM_HEAD == 4, "");
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    // each thread should load 8/(TILE_SZ_RATIO / TILE_NUM_HEAD) times per
    // iteration

    // INSERT KERNEL CODE HERE

    // int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;
    int ArowIdx = threadIdx.x / 32 * 16 + ((threadIdx.x % 32) < 16
                                               ? ((threadIdx.x % 32))
                                               : ((threadIdx.x % 32) - 16));
    int shdmemColIdxBias = (threadIdx.x % 32) < 16 ? 0 : 16;

    int shdmemLDBrowIdx =
        0 /*i*/ * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
    int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
    int shdmemLDBheadIdx =
        blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < K && shdmemLDBcolIdx < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, K, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx)
                 : 0.0f;
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < K && shdmemLDBcolIdx + 16 < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, K, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx + 16)
                 : 0.0f;

    __syncthreads();

    float reg0;
    float reg1;
    float reg2;
    float reg3;
    float reg4;
    float reg5;
    float reg6;
    float reg7;
    /*__shared__ float shmem_Adata[256*8];
    cg::thread_block_tile<32> tile32 =
    cg::tiled_partition<32>(cg::this_thread_block());
    // load A in registers; software pipelining
    cg::memcpy_async(tile32, shmem_Adata, &(A(0,0,0*8)), sizeof(float)*256*8);*/

    for (int i = 0; i < (K + 8 - 1) / (8); i++) {
      // shuffle: only half of the warp load the register
      // if (threadIdx.x%32<16){
      // TODO: async load A data to be used in the next iteration into the
      // shared memory
      /*cg::wait(tile32);
      reg0=shmem_Adata[ArowIdx+0*8];
      reg1=shmem_Adata[ArowIdx+1*8];
      reg2=shmem_Adata[ArowIdx+2*8];
      reg3=shmem_Adata[ArowIdx+3*8];
      reg4=shmem_Adata[ArowIdx+4*8];
      reg5=shmem_Adata[ArowIdx+5*8];
      reg6=shmem_Adata[ArowIdx+6*8];
      reg7=shmem_Adata[ArowIdx+7*8];

      __syncthreads();
      if (i<(k + 8 - 1) / (8) - 1){
          cg::memcpy_async(tile32, shmem_Adata, &(A(0,0,(i+1)*8)),
      sizeof(float)*256*8);
      }*/
      reg0 = (ArowIdx < OUT_DIM && K > i * 8)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8)
                 : 0.0f;
      reg1 = (ArowIdx < OUT_DIM && K > i * 8 + 1)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 1)
                 : 0.0f;
      reg2 = (ArowIdx < OUT_DIM && K > i * 8 + 2)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 2)
                 : 0.0f;
      reg3 = (ArowIdx < OUT_DIM && K > i * 8 + 3)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 3)
                 : 0.0f;
      reg4 = (ArowIdx < OUT_DIM && K > i * 8 + 4)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 4)
                 : 0.0f;
      reg5 = (ArowIdx < OUT_DIM && K > i * 8 + 5)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 5)
                 : 0.0f;
      reg6 = (ArowIdx < OUT_DIM && K > i * 8 + 6)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 6)
                 : 0.0f;
      reg7 = (ArowIdx < OUT_DIM && K > i * 8 + 7)
                 ? A(ArowIdx / K, ArowIdx % K, i * 8 + 7)
                 : 0.0f;

      // // shuffle: the second half of each warp get the register data from the
      // first half through warp shuffling reg0= __shfl_sync(0xffffffff, reg0,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg1=
      // __shfl_sync(0xffffffff, reg1,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg2=
      // __shfl_sync(0xffffffff, reg2,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg3=
      // __shfl_sync(0xffffffff, reg3,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg4=
      // __shfl_sync(0xffffffff, reg4,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg5=
      // __shfl_sync(0xffffffff, reg5,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg6=
      // __shfl_sync(0xffffffff, reg6,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16))); reg7=
      // __shfl_sync(0xffffffff, reg7,
      // ((threadIdx.x%32)<16?((threadIdx.x%32)):((threadIdx.x%32)-16)));

      // load B in shared memory
      // the loading scheme is adjusted to fit B's column-major layout
      int shdmemLDBrowIdx =
          i * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
      int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
      int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD +
                             threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);

      float next_iter_shmem_val_0 =
          (shdmemLDBrowIdx + 8 < K && shdmemLDBcolIdx < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, K, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx)
              : 0.0f;
      float next_iter_shmem_val_2 =
          (shdmemLDBrowIdx + 8 < K && shdmemLDBcolIdx + 16 < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, K, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx + 16)
              : 0.0f;

      // compute C
      if (ArowIdx < OUT_DIM) {
//// software pipelining to load data from shared memory for the next iteration
// float shmem_val0[2];
// float shmem_val1[2];
// float shmem_val2[2];
// float shmem_val3[2];
// float shmem_val4[2];
// float shmem_val5[2];
// float shmem_val6[2];
// float shmem_val7[2];
// shmem_val0[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 +
// shdmemColIdxBias][0]; shmem_val1[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A /
// TILE_NUM_HEAD)][0 + shdmemColIdxBias][1]; shmem_val2[0] = shmem[i %
// 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][2];
// shmem_val3[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 +
// shdmemColIdxBias][3]; shmem_val4[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A /
// TILE_NUM_HEAD)][0 + shdmemColIdxBias][4]; shmem_val5[0] = shmem[i %
// 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 + shdmemColIdxBias][5];
// shmem_val6[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0 +
// shdmemColIdxBias][6]; shmem_val7[0] = shmem[i % 2][threadIdx.x / (TILE_SZ_A /
// TILE_NUM_HEAD)][0 + shdmemColIdxBias][7];
#pragma unroll 2
        for (int shdmemColIdx = 0; shdmemColIdx < 16; shdmemColIdx++) {
          // software pipelining to load data from shared memory for the next
          // iteration if (shdmemColIdx < 15)
          //{
          //    shmem_val0[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x /
          //    (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 +
          //    shdmemColIdxBias][0]; shmem_val1[(shdmemColIdx + 1) % 2] =
          //    shmem[i % 2][threadIdx.x / (TILE_SZ_A /
          //    TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][1];
          //    shmem_val2[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x /
          //    (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 +
          //    shdmemColIdxBias][2]; shmem_val3[(shdmemColIdx + 1) % 2] =
          //    shmem[i % 2][threadIdx.x / (TILE_SZ_A /
          //    TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][3];
          //    shmem_val4[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x /
          //    (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 +
          //    shdmemColIdxBias][4]; shmem_val5[(shdmemColIdx + 1) % 2] =
          //    shmem[i % 2][threadIdx.x / (TILE_SZ_A /
          //    TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][5];
          //    shmem_val6[(shdmemColIdx + 1) % 2] = shmem[i % 2][threadIdx.x /
          //    (TILE_SZ_A / TILE_NUM_HEAD)][shdmemColIdx + 1 +
          //    shdmemColIdxBias][6]; shmem_val7[(shdmemColIdx + 1) % 2] =
          //    shmem[i % 2][threadIdx.x / (TILE_SZ_A /
          //    TILE_NUM_HEAD)][shdmemColIdx + 1 + shdmemColIdxBias][7];
          //}
          int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                        shdmemColIdxBias;

          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [0] /*shmem_val0[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [1] /*shmem_val1[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [2] /*shmem_val2[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [3] /*shmem_val3[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [4] /*shmem_val4[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [5] /*shmem_val5[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [6] /*shmem_val6[shdmemColIdx % 2]*/;
          /*C(ArowIdx / k, ArowIdx % k, CcolIdx)*/ shmem_output
              [shdmemColIdx][ArowIdx / 16][shdmemColIdxBias / 16]
              [ArowIdx % 16] +=
              reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                          [shdmemColIdx + shdmemColIdxBias]
                          [7] /*shmem_val7[shdmemColIdx % 2]*/;
        }
      }
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_0;
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_2;
      __syncthreads();
    }

    // TODO: optimize thread mapping to improve access pattern
    for (int store_iter = 0; store_iter < 256 * 32 / TILE_SZ_A; store_iter++) {
      int node_idx_1 = store_iter;
      int ele_idx_1 = ArowIdx / 16;
      int ele_idx_2 = ArowIdx % 16;
      int CcolIdx =
          store_iter + /*blockIdx.x * TILE_SZ_B*/ BcolBias + shdmemColIdxBias;
      if (CcolIdx < n) {
        GetCEle<C_col_scatter_flag, OUT_DIM>(
            C, C_col_scatter_list, K, ele_idx_1 * 16 / 64,
            ele_idx_2 + ele_idx_1 * 16 % 64,
            BcolBias + node_idx_1 + shdmemColIdxBias) =
            shmem_output[node_idx_1][ele_idx_1][shdmemColIdxBias / 16]
                        [ele_idx_2];
        // C(ArowIdx/k, ArowIdx%k, CcolIdx) =
        // shmem_output[shdmemColIdx][ArowIdx/16][shdmemColIdxBias/16][ArowIdx%16];
      }
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
  }
};

template <int OUT_DIM, int NUM_HEADS, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<256, 8, OUT_DIM, NUM_HEADS, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    constexpr int TILE_SZ_A = 256;
    constexpr int TILE_SZ_B = 8;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int TILE_NUM_HEAD = TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD;
    constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    assert(k == 64);
    assert(m == OUT_DIM);
/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1,
// ... -| |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest
// head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src head 1
// dest head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src
// head 1 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest head 0,
// src head 2 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest
// head 0, src head 2 dest head 1, ...  | |  src head 3 dest head 0, src head 3
// dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1,
// ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node
// 2 head 0, ... -| |  intermediate node 0 head 0, intermediate node 1 head 0,
// intermediate node 2 head 0, ...  | |  intermediate node 0 head 1,
// intermediate node 1 head 1, intermediate node 2 head 1, ...  | | intermediate
// node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node
// 2 head 2, ...  | |  intermediate node 0 head 2, intermediate node 1 head 2,
// intermediate node 2 head 2, ...  | |  intermediate node 0 head 3,
// intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node
// 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * k) + (row) + (col)*m]

    __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B]
                          [TILE_SZ_RATIO / TILE_NUM_HEAD];
    // TODO: change TILE_SZ_RATIO / TILE_NUM_HEAD to an individual value and
    // load multiple times to shared memory accordingly.

    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;

    int shdmemLDBrowIdx = 0 /*i*/ * TILE_SZ_RATIO / TILE_NUM_HEAD +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
                              (TILE_SZ_RATIO / TILE_NUM_HEAD);
    int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
                              (TILE_SZ_RATIO / TILE_NUM_HEAD);
    int shdmemLDBheadIdx =
        blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
          (TILE_SZ_RATIO / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
          (TILE_SZ_RATIO / TILE_NUM_HEAD)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx)
                 : 0.0f;
    __syncthreads();
    for (int i = 0; i < (k + TILE_SZ_RATIO / TILE_NUM_HEAD - 1) /
                            (TILE_SZ_RATIO / TILE_NUM_HEAD);
         i++) {
      // load A in registers
      float reg0 = 0.0f;
      float reg1 = 0.0f;
      float reg2 = 0.0f;
      float reg3 = 0.0f;
      float reg4 = 0.0f;
      float reg5 = 0.0f;
      float reg6 = 0.0f;
      float reg7 = 0.0f;
      if (ArowIdx < m) {
        reg0 =
            (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                : 0.0f;
        reg1 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   : 0.0f;
        reg2 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   : 0.0f;
        reg3 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   : 0.0f;
        reg4 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 4)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 4)
                   : 0.0f;
        reg5 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 5)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 5)
                   : 0.0f;
        reg6 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 6)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 6)
                   : 0.0f;
        reg7 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 7)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 7)
                   : 0.0f;
        /*reg4 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+4)?A(blockIdx.y*TILE_NUM_HEAD+
        (TILE_NUM_HEAD-1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg5 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+5)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg6 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+6)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg7 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+7)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;*/
      }
      // load B in shared memory
      // the loading scheme is adjusted to fit B's column-major layout
      int shdmemLDBrowIdx = i * TILE_SZ_RATIO / TILE_NUM_HEAD +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD +
                             threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);

      float next_iter_shmem_val =
          (shdmemLDBrowIdx + TILE_SZ_RATIO / TILE_NUM_HEAD < k &&
           shdmemLDBcolIdx < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + TILE_SZ_RATIO / TILE_NUM_HEAD,
                    shdmemLDBcolIdx)
              : 0.0f;

      // compute C
      if (ArowIdx < m) {
        for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++) {
          int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias;
          if (CcolIdx < n) {
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][0];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][1];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][2];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][3];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][4];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][5];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][6];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][7];
          }
        }
      }
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
            (TILE_SZ_RATIO / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
            (TILE_SZ_RATIO / TILE_NUM_HEAD)] = next_iter_shmem_val;
      __syncthreads();
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
  }
};

template <int OUT_DIM, int NUM_HEADS, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<256, 32, OUT_DIM, NUM_HEADS, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    constexpr int TILE_SZ_A = 256;
    constexpr int TILE_SZ_B = 32;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int TILE_NUM_HEAD = TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD;
    constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    assert(k == 64);
/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1,
// ... -| |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest
// head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src head 1
// dest head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src
// head 1 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest head 0,
// src head 2 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest
// head 0, src head 2 dest head 1, ...  | |  src head 3 dest head 0, src head 3
// dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1,
// ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node
// 2 head 0, ... -| |  intermediate node 0 head 0, intermediate node 1 head 0,
// intermediate node 2 head 0, ...  | |  intermediate node 0 head 1,
// intermediate node 1 head 1, intermediate node 2 head 1, ...  | | intermediate
// node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node
// 2 head 2, ...  | |  intermediate node 0 head 2, intermediate node 1 head 2,
// intermediate node 2 head 2, ...  | |  intermediate node 0 head 3,
// intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node
// 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * k) + (row) + (col)*m]

    __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B][8];
    static_assert(TILE_SZ_RATIO / TILE_NUM_HEAD == 2, "");
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    // each thread should load 8/(TILE_SZ_RATIO / TILE_NUM_HEAD) times per
    // iteration

    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;

    int shdmemLDBrowIdx =
        0 /*i*/ * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
    int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
    int shdmemLDBheadIdx =
        blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx)
                 : 0.0f;
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 8]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx + 8 < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx + 8)
                 : 0.0f;
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx + 16 < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx + 16)
                 : 0.0f;
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 24]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx + 24 < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx + 24)
                 : 0.0f;

    __syncthreads();
    for (int i = 0; i < (k + 8 - 1) / (8); i++) {
      // load A in registers
      float reg0 = 0.0f;
      float reg1 = 0.0f;
      float reg2 = 0.0f;
      float reg3 = 0.0f;
      float reg4 = 0.0f;
      float reg5 = 0.0f;
      float reg6 = 0.0f;
      float reg7 = 0.0f;
      if (ArowIdx < m) {
        reg0 = (k > i * 8) ? A(ArowIdx / k, ArowIdx % k, i * 8) : 0.0f;
        reg1 = (k > i * 8 + 1) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 1) : 0.0f;
        reg2 = (k > i * 8 + 2) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 2) : 0.0f;
        reg3 = (k > i * 8 + 3) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 3) : 0.0f;
        reg4 = (k > i * 8 + 4) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 4) : 0.0f;
        reg5 = (k > i * 8 + 5) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 5) : 0.0f;
        reg6 = (k > i * 8 + 6) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 6) : 0.0f;
        reg7 = (k > i * 8 + 7) ? A(ArowIdx / k, ArowIdx % k, i * 8 + 7) : 0.0f;
        /*reg4 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+4)?A(blockIdx.y*TILE_NUM_HEAD+
        (TILE_NUM_HEAD-1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg5 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+5)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg6 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+6)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg7 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+7)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;*/
      }
      // load B in shared memory
      // the loading scheme is adjusted to fit B's column-major layout
      int shdmemLDBrowIdx =
          i * 8 + (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8);
      int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8);
      int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD +
                             threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);

      float next_iter_shmem_val_0 =
          (shdmemLDBrowIdx + 8 < k && shdmemLDBcolIdx < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx)
              : 0.0f;
      float next_iter_shmem_val_1 =
          (shdmemLDBrowIdx + 8 < k && shdmemLDBcolIdx + 8 < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx + 8)
              : 0.0f;
      float next_iter_shmem_val_2 =
          (shdmemLDBrowIdx + 8 < k && shdmemLDBcolIdx + 16 < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx + 16)
              : 0.0f;
      float next_iter_shmem_val_3 =
          (shdmemLDBrowIdx + 8 < k && shdmemLDBcolIdx + 24 < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + 8, shdmemLDBcolIdx + 24)
              : 0.0f;

      // compute C
      if (ArowIdx < m) {
        for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++) {
          int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias;
          if (CcolIdx < n) {
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][0];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][1];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][2];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][3];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][4];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][5];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][6];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][7];
          }
        }
      }
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_0;
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 8]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_1;
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 16]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_2;
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) / (8) + 24]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) % (8)] =
               next_iter_shmem_val_3;
      __syncthreads();
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
  }
};

template <int OUT_DIM, int NUM_HEADS, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<128, 16, OUT_DIM, NUM_HEADS, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    constexpr int TILE_SZ_A = 128;
    constexpr int TILE_SZ_B = 16;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int TILE_NUM_HEAD = TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD;
    constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    assert(k == 64);
/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1,
// ... -| |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest
// head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src head 1
// dest head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src
// head 1 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest head 0,
// src head 2 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest
// head 0, src head 2 dest head 1, ...  | |  src head 3 dest head 0, src head 3
// dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1,
// ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node
// 2 head 0, ... -| |  intermediate node 0 head 0, intermediate node 1 head 0,
// intermediate node 2 head 0, ...  | |  intermediate node 0 head 1,
// intermediate node 1 head 1, intermediate node 2 head 1, ...  | | intermediate
// node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node
// 2 head 2, ...  | |  intermediate node 0 head 2, intermediate node 1 head 2,
// intermediate node 2 head 2, ...  | |  intermediate node 0 head 3,
// intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node
// 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * k) + (row) + (col)*m]
    __shared__ float shmem[TILE_NUM_HEAD][TILE_SZ_RATIO / TILE_NUM_HEAD]
                          [TILE_SZ_B];

    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;

    for (int i = 0; i < (k + TILE_SZ_RATIO / TILE_NUM_HEAD - 1) /
                            (TILE_SZ_RATIO / TILE_NUM_HEAD);
         i++) {
      // load A in registers
      float reg0 = 0.0f;
      float reg1 = 0.0f;
      float reg2 = 0.0f;
      float reg3 = 0.0f;
      /*float reg4=0.0f;
      float reg5=0.0f;
      float reg6=0.0f;
      float reg7=0.0f;*/
      if (ArowIdx < m) {
        reg0 =
            (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                : 0.0f;
        reg1 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   : 0.0f;
        reg2 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   : 0.0f;
        reg3 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   : 0.0f;
        /*reg4 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+4)?A(blockIdx.y*TILE_NUM_HEAD+
        (TILE_NUM_HEAD-1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg5 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+5)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg6 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+6)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg7 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+7)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;*/
      }
      // load B in shared memory
      // the loading scheme is adjusted to fit B's column-major layout
      int shdmemLDBrowIdx = i * TILE_SZ_RATIO / TILE_NUM_HEAD +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD +
                             threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
      shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
            (TILE_SZ_RATIO / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
            (TILE_SZ_RATIO / TILE_NUM_HEAD)] =
               (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n)
                   ? GetBEle<
                         B_col_gather_flag, OUT_DIM,
                         B_col_second_indirection_gather_flag,
                         B_col_second_indirection_gather_binary_search_flag>(
                         B, B_col_gather_list, B_col_second_gather_list,
                         B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                         shdmemLDBrowIdx, shdmemLDBcolIdx)
                   : 0.0f;

      __syncthreads();
      // compute C
      if (ArowIdx < m) {
        for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++) {
          int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias;
          if (CcolIdx < n) {
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg0 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][0]
                            [shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg1 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][1]
                            [shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg2 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][2]
                            [shdmemColIdx];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg3 * shmem[threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)][3]
                            [shdmemColIdx];
            /*C(ArowIdx / k, ArowIdx % k, CcolIdx)+=reg4*shmem[1-threadIdx.x /
            (TILE_SZ_A / TILE_NUM_HEAD)][0][shdmemColIdx]; C(ArowIdx / k,
            ArowIdx % k, CcolIdx)+=reg5*shmem[1-threadIdx.x / (TILE_SZ_A /
            TILE_NUM_HEAD)][1][shdmemColIdx]; C(ArowIdx / k, ArowIdx % k,
            CcolIdx)+=reg6*shmem[1-threadIdx.x / (TILE_SZ_A /
            TILE_NUM_HEAD)][2][shdmemColIdx]; C(ArowIdx / k, ArowIdx % k,
            CcolIdx)+=reg7*shmem[1-threadIdx.x / (TILE_SZ_A /
            TILE_NUM_HEAD)][3][shdmemColIdx];*/
          }
        }
      }
      __syncthreads();
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
  }
};

template <int OUT_DIM, int NUM_HEADS, bool B_col_gather_flag,
          bool C_col_scatter_flag, bool B_col_second_indirection_gather_flag,
          bool B_col_second_indirection_gather_binary_search_flag>
class mysgemm_functor<128, 8, OUT_DIM, NUM_HEADS, B_col_gather_flag,
                      C_col_scatter_flag, B_col_second_indirection_gather_flag,
                      B_col_second_indirection_gather_binary_search_flag> {
 public:
  __device__ __forceinline__ static void exec_function(
      int m, int n, int k, float *A, float *B, float *C, int *B_col_gather_list,
      int *B_col_second_gather_list, int B_col_second_gather_list_length,
      int *C_col_scatter_list, int BcolBias) {
    constexpr int TILE_SZ_A = 128;
    constexpr int TILE_SZ_B = 8;
    constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
    constexpr int TILE_NUM_HEAD = TILE_SZ_A / NODE_INPUT_DIM_PER_HEAD;
    constexpr int TILE_SZ_RATIO = TILE_SZ_A / TILE_SZ_B;
    static_assert(TILE_SZ_RATIO % TILE_NUM_HEAD == 0, "");
    assert(k == 64);
/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is column major as well
 * m == 256, n == number of unique dest nodes in this relation, k == 64
 * m stands for OUT_DIM, k stands for NODE_INPUT_DIM_PER_HEAD
 ********************************************************************/
// layout of B (column major)
// |- node 0 head 0, node 1 head 0, node 2 head 0, ... -|
// |  node 0 head 0, node 1 head 0, node 2 head 0, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 1, node 1 head 1, node 2 head 1, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 2, node 1 head 2, node 2 head 2, ...  |
// |  node 0 head 3, node 1 head 3, node 2 head 3, ...  |
// |- node 0 head 3, node 1 head 3, node 2 head 3, ... -|

// layout of A (column major)
// |- src head 0 dest head 0, src head 0 dest head 0, src head 0 dest head 1,
// ... -| |  src head 0 dest head 0, src head 0 dest head 0, src head 0 dest
// head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src head 1
// dest head 1, ...  | |  src head 1 dest head 0, src head 1 dest head 0, src
// head 1 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest head 0,
// src head 2 dest head 1, ...  | |  src head 2 dest head 0, src head 2 dest
// head 0, src head 2 dest head 1, ...  | |  src head 3 dest head 0, src head 3
// dest head 0, src head 3 dest head 1, ...  |
// |- src head 3 dest head 0, src head 3 dest head 0, src head 3 dest head 1,
// ... -|

// layout of C (column major)
// |- intermediate node 0 head 0, intermediate node 1 head 0, intermediate node
// 2 head 0, ... -| |  intermediate node 0 head 0, intermediate node 1 head 0,
// intermediate node 2 head 0, ...  | |  intermediate node 0 head 1,
// intermediate node 1 head 1, intermediate node 2 head 1, ...  | | intermediate
// node 0 head 1, intermediate node 1 head 1, intermediate node 2 head 1, ...  |
// |  intermediate node 0 head 2, intermediate node 1 head 2, intermediate node
// 2 head 2, ...  | |  intermediate node 0 head 2, intermediate node 1 head 2,
// intermediate node 2 head 2, ...  | |  intermediate node 0 head 3,
// intermediate node 1 head 3, intermediate node 2 head 3, ...  |
// |- intermediate node 0 head 3, intermediate node 1 head 3, intermediate node
// 2 head 3, ... -|

// Macros for accessing flattened matrices
#define A(idx_head, row, col) A[(idx_head * k) + (row) + (col)*m]
    __shared__ float shmem[2 /*double buffering*/][TILE_NUM_HEAD][TILE_SZ_B]
                          [TILE_SZ_RATIO / TILE_NUM_HEAD];

    // INSERT KERNEL CODE HERE

    int ArowIdx = blockIdx.y * TILE_SZ_A + threadIdx.x;

    int shdmemLDBrowIdx = 0 /*i*/ * TILE_SZ_RATIO / TILE_NUM_HEAD +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
                              (TILE_SZ_RATIO / TILE_NUM_HEAD);
    int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                          (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
                              (TILE_SZ_RATIO / TILE_NUM_HEAD);
    int shdmemLDBheadIdx =
        blockIdx.y * TILE_NUM_HEAD + threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);
    shmem[0][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
          (TILE_SZ_RATIO / TILE_NUM_HEAD)]
         [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
          (TILE_SZ_RATIO / TILE_NUM_HEAD)] =
             (shdmemLDBrowIdx < k && shdmemLDBcolIdx < n)
                 ? GetBEle<B_col_gather_flag, OUT_DIM,
                           B_col_second_indirection_gather_flag,
                           B_col_second_indirection_gather_binary_search_flag>(
                       B, B_col_gather_list, B_col_second_gather_list,
                       B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                       shdmemLDBrowIdx, shdmemLDBcolIdx)
                 : 0.0f;
    __syncthreads();
    for (int i = 0; i < (k + TILE_SZ_RATIO / TILE_NUM_HEAD - 1) /
                            (TILE_SZ_RATIO / TILE_NUM_HEAD);
         i++) {
      // load A in registers
      float reg0 = 0.0f;
      float reg1 = 0.0f;
      float reg2 = 0.0f;
      float reg3 = 0.0f;
      float reg4 = 0.0f;
      float reg5 = 0.0f;
      float reg6 = 0.0f;
      float reg7 = 0.0f;
      if (ArowIdx < m) {
        reg0 =
            (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                ? A(ArowIdx / k, ArowIdx % k, i * TILE_SZ_RATIO / TILE_NUM_HEAD)
                : 0.0f;
        reg1 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 1)
                   : 0.0f;
        reg2 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 2)
                   : 0.0f;
        reg3 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 3)
                   : 0.0f;
        reg4 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 4)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 4)
                   : 0.0f;
        reg5 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 5)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 5)
                   : 0.0f;
        reg6 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 6)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 6)
                   : 0.0f;
        reg7 = (k > i * TILE_SZ_RATIO / TILE_NUM_HEAD + 7)
                   ? A(ArowIdx / k, ArowIdx % k,
                       i * TILE_SZ_RATIO / TILE_NUM_HEAD + 7)
                   : 0.0f;
        /*reg4 = (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+4)?A(blockIdx.y*TILE_NUM_HEAD+
        (TILE_NUM_HEAD-1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg5 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+5)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg6 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+6)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f; reg7 =
        (k>i*TILE_SZ_RATIO/TILE_NUM_HEAD+7)?A(blockIdx.y * TILE_NUM_HEAD +
        (TILE_NUM_HEAD - 1 - threadIdx.x / k), ArowIdx %
        k,i*TILE_SZ_RATIO/TILE_NUM_HEAD):0.0f;*/
      }
      // load B in shared memory
      // the loading scheme is adjusted to fit B's column-major layout
      int shdmemLDBrowIdx = i * TILE_SZ_RATIO / TILE_NUM_HEAD +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBcolIdx = /*blockIdx.x * TILE_SZ_B*/ BcolBias +
                            (threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
                                (TILE_SZ_RATIO / TILE_NUM_HEAD);
      int shdmemLDBheadIdx = blockIdx.y * TILE_NUM_HEAD +
                             threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD);

      float next_iter_shmem_val =
          (shdmemLDBrowIdx + TILE_SZ_RATIO / TILE_NUM_HEAD < k &&
           shdmemLDBcolIdx < n)
              ? GetBEle<B_col_gather_flag, OUT_DIM,
                        B_col_second_indirection_gather_flag,
                        B_col_second_indirection_gather_binary_search_flag>(
                    B, B_col_gather_list, B_col_second_gather_list,
                    B_col_second_gather_list_length, k, shdmemLDBheadIdx,
                    shdmemLDBrowIdx + TILE_SZ_RATIO / TILE_NUM_HEAD,
                    shdmemLDBcolIdx)
              : 0.0f;

      // compute C
      if (ArowIdx < m) {
        for (int shdmemColIdx = 0; shdmemColIdx < TILE_SZ_B; shdmemColIdx++) {
          int CcolIdx = shdmemColIdx + /*blockIdx.x * TILE_SZ_B*/ BcolBias;
          if (CcolIdx < n) {
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg0 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][0];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg1 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][1];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg2 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][2];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg3 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][3];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg4 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][4];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg5 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][5];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg6 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][6];
            GetCEle<C_col_scatter_flag, OUT_DIM>(
                C, C_col_scatter_list, k, ArowIdx / k, ArowIdx % k, CcolIdx) +=
                reg7 * shmem[i % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
                            [shdmemColIdx][7];
          }
        }
      }
      shmem[(i + 1) % 2][threadIdx.x / (TILE_SZ_A / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) /
            (TILE_SZ_RATIO / TILE_NUM_HEAD)]
           [(threadIdx.x % (TILE_SZ_A / TILE_NUM_HEAD)) %
            (TILE_SZ_RATIO / TILE_NUM_HEAD)] = next_iter_shmem_val;
      __syncthreads();
    }

    // SSL Hint (9/6/21): try using just one register for the tile of A
    // rather than several--in other words, load one value (per thread)
    // from A and compute using that value rather than loading all values
    // before doing the computation.  This approach seems to be slightly
    // faster than the alternative.
#undef A
    // refactor the following 4 macros soly used in this kernel into constexprs
    // expressed in template integers
    //#undef TILE_NUM_HEAD
    //#undef TILE_SZ_RATIO
    //#undef TILE_SZ_B
    //#undef TILE_SZ_A
  }
};