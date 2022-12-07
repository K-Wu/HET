#pragma once
#include <cuda_runtime.h>
#include "cuda.h"
#include "utils.cu.h"

// code from
// http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
//@@ Example of grid and block configuration
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// dim3 dimGrid(B.width / dimBlock.x, A.height /
// dimBlock.y, num_heads * num_partitions_along_Acol_Brow );
// MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
// C = A * B (all are row-major matrices)
template <int BLOCK_SIZE, typename Idx, typename IdxPtr,
          bool hgt_instead_of_rgcn_flag>
__device__ __forceinline__ void _simplified_basic_MatMulKernel(
    float* A, float* B, float* C, float* edge_norm, /*float* relation_pri,*/
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, Idx idx_relation, Idx numARows,
    Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
    Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, Idx num_heads) {
  // num_B_cols is output_dim//num_heads as forward propagation weight,
  // output_dim//num_heads as backward propagation weight, and in_feat_dim as
  // features or delta features. num_A_cols is input_dim as forward propagation
  // input feature, output_dim//num_heads as delta input feature

  // NB: when OuterProductFlag is true, num_heads of the input features is
  // always 1 and the other is num_heads, at least in case of RGAT. In the
  // implementation, when OuterProductFlag is true, A is always input and B the
  // gradient output feature. It is safe to pass num_A_cols as in_dim.

  // Block row and column
  if constexpr (!hgt_instead_of_rgcn_flag) {
    // assuming this case is RGCN and there is no multiple head
    assert((blockDim.z == 1));
  }  // otherwise assuming HGT
  Idx idx_head = blockIdx.z;
  IdxPtr A_gather_list = separate_coo_row_idx;
  IdxPtr C_scatter_list = separate_coo_col_idx;

  // Idx blockRow = blockIdx.y - blockIdxAlongRowBeg;
  Idx blockFeat = blockIdx.x;  // when OuterProductFlag==True, it is in [0,
                               // output_dim//num_heads)

  Idx blockRowLoopBeg, blockRowLoopEnd, blockRowLoopInc;

  blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
  blockRowLoopEnd = ceil_div<>(numARows, (int64_t)BLOCK_SIZE);
  blockRowLoopInc = strideNumBlocksAlongRow;

  for (Idx blockRow = blockRowLoopBeg; blockRow < blockRowLoopEnd;
       blockRow += blockRowLoopInc) {
    // NB: blockTask == blockIdx.x / ceil_div( num_B_cols, BLOCK_SIZE)

    // Each thread block computes one sub-matrix Csub of C
    // float* Csub = &C[blockRow * BLOCK_SIZE * num_B_cols + blockFeat *
    // BLOCK_SIZE]; Each thread computes one element of Csub by accumulating
    // results into Cvalue
    float Cvalue = 0.0f;
    // Thread row and column within Csub
    Idx thIdxRow = threadIdx.y;
    Idx thIdxFeat = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results

    Idx mLoopBeg, mLoopEnd, mLoopInc;

    mLoopBeg = ceil_div<Idx>(num_A_cols, BLOCK_SIZE);
    mLoopEnd = ceil_div<Idx>(num_A_cols, BLOCK_SIZE);
    mLoopInc = 1;

    for (Idx m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
      // Shared memory used to store Asub and Bsub respectively
      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

      // Get sub-matrix Bsub of B
      // float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockFeat * BLOCK_SIZE];
      // float* Asub;
      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix

      // Get sub-matrix Asub of A
      // Asub = &A[blockRow * BLOCK_SIZE * num_A_cols + m * BLOCK_SIZE];

      As[thIdxRow][thIdxFeat] =
          (thIdxRow + blockRow * BLOCK_SIZE < numARows &&
           m * BLOCK_SIZE + thIdxFeat < num_A_cols)
              ? A[A_gather_list[thIdxRow + blockRow * BLOCK_SIZE +
                                blockRowJobEntryBeg] *
                      num_A_cols * num_heads +
                  num_A_cols * idx_head + (thIdxFeat + m * BLOCK_SIZE)] *
                    edge_norm[separate_coo_eids[thIdxRow +
                                                blockRow * BLOCK_SIZE +
                                                blockRowJobEntryBeg]]
              : 0.0f;
      // if constexpr (hgt_instead_of_rgcn_flag) {
      //   As[thIdxRow][thIdxFeat] *=
      //       relation_pri[idx_relation * num_heads + idx_head];
      // }

      // B matrix the most major dimension is num_heads, i.e., [num_heads,
      // num_B_rows_feat, num_B_cols_feat] instead of [num_nodes|edges,
      // num_heads, num_feats]
      // NB: B's num_B_cols_feat is the same as input_dim whereas
      // num_B_rows_feat is per head i.e., B dimension is [num_heads,
      // input_dim (num_A_cols), output_dim//num_heads (num_B_cols)] in
      // forward propagation or [num_heads, output_dim//num_heads
      // (num_A_cols), input_dim (num_B_cols)] in backward propagation
      // FIXME: incorporate num_head_one_flag

      Bs[thIdxRow][thIdxFeat] =
          (m * BLOCK_SIZE + thIdxRow < num_A_cols &&
           blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols)
              ? B[(m * BLOCK_SIZE + thIdxRow) * num_B_cols * num_heads +
                  idx_head * num_B_cols + (blockFeat * BLOCK_SIZE + thIdxFeat)]
              : 0.0f;

      // if (ScatterCFlag && !AdvancedScatterCFlag) {
      //   bool WriteCInRangeFlag =
      //       thIdxRow + blockRow * BLOCK_SIZE < numARows &&
      //       idx_head < num_heads &&
      //       blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols;
      //   if (C_scatter_list[thIdxRow + blockRow * BLOCK_SIZE +
      //                      blockRowJobEntryBeg] == 0 &&
      //       WriteCInRangeFlag) {
      //     bool bflag = m * BLOCK_SIZE + thIdxRow < num_A_cols &&
      //                  blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols &&
      //                  idx_head < num_heads;
      //     bool aflag = thIdxRow + blockRow * BLOCK_SIZE < numARows &&
      //                  m * BLOCK_SIZE + thIdxFeat < num_A_cols &&
      //                  idx_head < num_heads;

      //     printf(
      //         "0 found(WriteCInRangeFlag)!!! (thIdxRow %ld, blockRow %ld, "
      //         "blockRowJobEntryBeg "
      //         "%ld, numARows %ld), (thIdxFeat %ld, blockFeat %ld,
      //         num_B_cols "
      //         "%ld), (idx_head %ld, num_head %ld) (idx_relation %ld) "
      //         "(mLoopBeg %ld mLoopEnd %ld m%ld) (bweightflag %d aflag %d "
      //         "bflag %d) (shmem A %f B %f)\n",
      //         thIdxRow, blockRow, blockRowJobEntryBeg, numARows, thIdxFeat,
      //         blockFeat, num_B_cols, idx_head, num_heads, idx_relation,
      //         mLoopBeg, mLoopEnd, m, BWeightInsteadOfFeatureFlag, aflag,
      //         bflag, As[thIdxRow][thIdxFeat], Bs[thIdxRow][thIdxFeat]);
      //   }
      // }

      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation
      __syncthreads();
      // Multiply Asub and Bsub together
      for (int e = 0; e < BLOCK_SIZE; ++e)
        Cvalue += As[thIdxRow][e] * Bs[e][thIdxFeat];
      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element

    bool WriteCInRangeFlag = thIdxRow + blockRow * BLOCK_SIZE < numARows &&
                             blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols;
    // if (ScatterCFlag && !AdvancedScatterCFlag) {
    //   if (C_scatter_list[thIdxRow + blockRow * BLOCK_SIZE +
    //                      blockRowJobEntryBeg] == 0) {
    //     printf(
    //         "0 found(%d)!!! (thIdxRow %ld, blockRow %ld,
    //         blockRowJobEntryBeg "
    //         "%ld, numARows %ld), (thIdxFeat %ld, blockFeat %ld, num_B_cols
    //         "
    //         "%ld), (idx_head %ld, num_head %ld) (idx_relation %ld)
    //         (mLoopBeg "
    //         "%ld mLoopEnd %ld) CValue %f\n",
    //         WriteCInRangeFlag, thIdxRow, blockRow, blockRowJobEntryBeg,
    //         numARows, thIdxFeat, blockFeat, num_B_cols, idx_head,
    //         num_heads, idx_relation, mLoopBeg, mLoopEnd, Cvalue);
    //   }
    // }
    if (WriteCInRangeFlag) {
      // print GetRowMajorElement arguments
      // if constexpr (!OuterProductFlag && !GatherAFlag &&
      // !AdvancedGatherAFlag && !GatherBFlag && !AdvancedGatherBFlag &&
      // ScatterCFlag && !AdvancedScatterCFlag && AtomicUpdateFlag)
      // if (ScatterCFlag && !AdvancedScatterCFlag)
      // printf("C %p C_scatter_list %p unique_srcs_and_dests_rel_ptr %p "
      //        "unique_srcs_and_dests_node_indices %p idx_relation %ld "
      //        "idxRow %ld scatter_list[idxRow] %ld idxFeat %ld idx_head
      //        %ld num_heads %ld " "num_B_cols %ld, numARows %ld\n", C,
      //        C_scatter_list, unique_srcs_and_dests_rel_ptr,
      //        unique_srcs_and_dests_node_indices, idx_relation,
      //        thIdxRow + blockRow * BLOCK_SIZE, C_scatter_list[thIdxRow +
      //        blockRow * BLOCK_SIZE], blockFeat * BLOCK_SIZE + thIdxFeat,
      //        idx_head, num_heads, num_B_cols, numARows);

      atomicAdd(&C[C_scatter_list[thIdxRow + blockRow * BLOCK_SIZE +
                                  blockRowJobEntryBeg] *
                       num_B_cols * num_heads +
                   idx_head * num_B_cols + blockFeat * BLOCK_SIZE + thIdxFeat],
                Cvalue);

      // atomicAdd(&GetRowMajorElement<Idx, IdxPtr, ScatterCFlag,
      //                               AdvancedScatterCFlag>(
      //               C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
      //               unique_srcs_and_dests_node_indices, idx_relation,
      //               thIdxRow + blockRow * BLOCK_SIZE + blockRowJobEntryBeg,
      //               C_num_head_one_flag ? 0 : idx_head,
      //               blockFeat * BLOCK_SIZE + thIdxFeat,
      //               C_num_head_one_flag ? 1 : num_heads, num_B_cols),
      //           Cvalue);
    }
  }
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGCNMatmulNoScatterGatherListFwProp(
    float* node_feat_input, float* weights, float* linear_projected_node_feat,
    float* edge_norm, IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int* accum_num_blocks_per_relation, Idx num_relations, Idx input_dim,
    Idx output_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<BLOCK_SIZE, Idx, IdxPtr, false>(
      node_feat_input, &weights[idx_relation * input_dim * output_dim],
      linear_projected_node_feat, edge_norm, /*nullptr, */ separate_coo_row_idx,
      separate_coo_col_idx, separate_coo_eids, idx_relation,
      separate_coo_rel_ptrs[idx_relation + 1] -
          separate_coo_rel_ptrs[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      separate_coo_rel_ptrs[idx_relation], input_dim, output_dim, 1);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HGTMessageGenerationAndAccumulationFwProp(
    float* node_feat_input, float* weights, float* linear_projected_node_feat,
    float* edge_norm, /*float* relation_pri, */ IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx input_dim, Idx output_dim, Idx num_heads) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<BLOCK_SIZE, Idx, IdxPtr, true>(
      node_feat_input, &weights[idx_relation * input_dim * output_dim],
      linear_projected_node_feat, edge_norm,
      /*relation_pri, */ separate_coo_row_idx, separate_coo_col_idx,
      separate_coo_eids, idx_relation,
      separate_coo_rel_ptrs[idx_relation + 1] -
          separate_coo_rel_ptrs[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      separate_coo_rel_ptrs[idx_relation], input_dim, output_dim, num_heads);
}