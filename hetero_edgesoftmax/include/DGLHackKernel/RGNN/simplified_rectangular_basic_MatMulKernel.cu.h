/*
    1/28: Jason Yan created this code on 2/1 when he was not really sure what he
   is doing 2/5: Don't worry about this file for now, the updated function is in
   UtilityAndPlayground at line 130 to line 280
*/

#pragma once
#include <cuda_runtime.h>

#include "utils.cu.h"

template <int SHMEM_BLOCKSIZE, typename Idx, typename IdxPtr>
class _simplified_rectangular_basic_MatMulKernel<SHMEM_BLOCKSIZE, Idx, IdxPtr> {
 public:
  __device__ __forceinline__ static void execute_function(
      float *A, float *B, float *C, float *edge_norm, float *inner_product,
      float *input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
      IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids, Idx idx_relation,
      Idx numARows, Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    {
      /*
        I put all the variable in the argument because if this is the case,
        I should be able to load a normal file and test if my algorithm works
        unless there are other things to change, which there probably are
      */
      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int blockFeat = blockIdx.x;

      int blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
      int blockRowLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE);
      int blockRowLoopInc = strideNumBlocksAlongRow;

      int Row = thIdRow + blockRow * SHMEM_BLOCK_SIZE + blockRowJobEntryBeg;
      int Col =
          idx_head * num_B_cols + blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat;

      // Create submatrix Asub and Bsub in shared memory
      __shared__ float As[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];
      __shared__ float Bs[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];

      for (int blockRow = blockRowLoopBeg; blockRow < blockRowLoopEnd;
           blockRow += blockRowLoopInc) {
        float Cvalue = 0.0f;  // If I use 0.0, it will be a double
        int thIdxRow_initial = threadIdx.y;
        int thIdxFeat_initial = threadIdx.x;

        int mLoopBeg = 0;
        int mLoopEnd = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE);
        int mLoopInc = 1;

        for (int m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
          // Load matrix code here
          As[ty][tx] = A[Row * num_B_cols + m * SHMEM_BLOCK_SIZE + tx];
          Bs[ty][tx] = B[(m * SHMEM_BLOCK_SIZE + ty) * Width + Col];

          // Matrix Sum Code
          for (int e = 0; e < SHMEM_BLOCK_SIZE; e++) {
            Cvalue += As[thIdRow_initial][e] * Bs[e][thIdxFeat_initial];
          }

          __syncthreads();
          // Synchronize to make sure all matrix mul are completed
        }

        __syncthreads();
        // Synchronize to make sure all sub-matrices are loaded

        C[Row * num_B_cols * num_heads + Col] = Cvalue;
      }
    }
  }
}