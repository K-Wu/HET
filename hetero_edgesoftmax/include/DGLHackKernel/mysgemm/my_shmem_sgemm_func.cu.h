#pragma once
#include <cuda_runtime.h>
#include "cuda.h"
#include "utils.cu.h"

template <bool GatherFlag, typename Idx, typename IdxPtr>
__device__ __forceinline__ float& GetRowMajorElementBasic(
    float* matrix_data, IdxPtr gather_list, Idx num_heads,
    Idx feat_dim_per_head, Idx row, Idx idx_head, Idx idx_feat) {
  if constexpr (GatherFlag) {
    return matrix_data[idx_head * feat_dim_per_head +
                       gather_list[row] * num_heads * feat_dim_per_head +
                       idx_feat];
  } else {
    return matrix_data[idx_head * feat_dim_per_head +
                       row * num_heads * feat_dim_per_head + idx_feat];
  }
}

template <typename Idx, typename IdxPtr>
__device__ __forceinline__ float& GetRowMajorElementAdvanced(
    float* matrix_data, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx idx_node,
    Idx idx_head, Idx idx_feat, Idx num_heads, Idx feat_dim_per_head) {
  Idx offset = find_relational_compact_as_of_node_index(
      idx_relation, idx_node, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices);
  return GetRowMajorElementBasic<false, Idx, IdxPtr>(
      matrix_data, nullptr, num_heads, feat_dim_per_head, offset, idx_head,
      idx_feat);
}

template <typename Idx, typename IdxPtr, bool GatherScatterFlag,
          bool AdvancedGatherScatterFlag>
__device__ __forceinline__ float& GetRowMajorElement(
    float* matrix_data, IdxPtr gather_scatter_list,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx idx_row,
    Idx idx_head, Idx idx_feat, Idx num_heads, Idx feat_dim_per_head) {
  if constexpr (GatherScatterFlag && AdvancedGatherScatterFlag) {
    Idx idx_node = gather_scatter_list[idx_row];
    return GetRowMajorElementAdvanced<Idx, IdxPtr>(
        matrix_data, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, idx_relation, idx_node, idx_head,
        idx_feat, num_heads, feat_dim_per_head);
  } else {
    return GetRowMajorElementBasic<GatherScatterFlag, Idx, IdxPtr>(
        matrix_data, gather_scatter_list, num_heads, feat_dim_per_head, idx_row,
        idx_head, idx_feat);
  }
}

// code from
// http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
//@@ Example of grid and block configuration
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// dim3 dimGrid(B.width / dimBlock.x, A.height /
// dimBlock.y, num_heads * num_partitions_along_Acol_Brow );
// MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
// C = A * B (all are row-major matrices)
template <int BLOCK_SIZE, bool OuterProductFlag, bool GatherAFlag,
          bool AdvancedGatherAFlag, bool GatherBFlag, bool AdvancedGatherBFlag,
          bool ScatterCFlag, bool AdvancedScatterCFlag, bool AtomicUpdateFlag,
          typename Idx, typename IdxPtr>
__device__ __forceinline__ void _basic_MatMulKernel(
    float* A, float* B, float* C, IdxPtr A_gather_list, IdxPtr B_gather_list,
    IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx numARows,
    Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
    Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, Idx num_heads) {
  // num_B_cols is output_dim//num_heads as forward propagation weight,
  // output_dim//num_heads as backward propagation weight, and feat_dim as
  // features or delta features num_A_cols is input_dim as forward propagation
  // input feature, input_dim as delta output feature
  constexpr bool BWeightInsteadOfFeatureFlag = !OuterProductFlag;

  // NB: when OuterProductFlag is true, num_heads of the input features is
  // always 1 and the other is num_heads, at least in case of RGAT. In the
  // implementation, when OuterProductFlag is true, A is always input and B the
  // gradient output feature. It is safe to pass num_A_cols as in_dim.

  // Block row and column
  Idx idx_head = blockIdx.z % num_heads;

  assert(NumInnerProductionPartitions > 0);
  if constexpr (OuterProductFlag) {
    CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(OuterProductFlag, AtomicUpdateFlag, "");
  } else {
    assert((blockDim.z == num_heads));
  }

  // Idx blockRow = blockIdx.y - blockIdxAlongRowBeg;
  Idx blockFeat = blockIdx.x;  // when OuterProductFlag==True, it is in [0,
                               // output_dim//num_heads)

  Idx blockRowLoopBeg, blockRowLoopEnd, blockRowLoopInc;
  if constexpr (OuterProductFlag) {
    blockRowLoopBeg = blockIdx.y;  // [0, input_dim)
    blockRowLoopEnd = blockIdx.y + 1;
    blockRowLoopInc = 1;
  } else {
    blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
    blockRowLoopEnd = ceil_div<>(numARows, (int64_t)BLOCK_SIZE);
    blockRowLoopInc = strideNumBlocksAlongRow;
  }

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
    if constexpr (OuterProductFlag) {
      // the block configuration scheme is different when OuterProductFlag
      // compared with !OuterProductFlag case when OuterProductFlag, the block
      // configuration is (num_output_per_head_dim//BLOCK_SIZE,
      // num_input_dim//BLOCK_SIZE,num_heads * num_edges) when
      // !OuterProductFlag, the block configuration is
      // (num_output_per_head_dim//BLOCK_SIZE,num_edges,num_heads)
      Idx blockAssignmentIdx = blockIdx.z / num_heads;
      mLoopBeg = blockAssignmentIdx - blockIdxAlongRowBeg;
      mLoopEnd = ceil_div<>(numARows, (int64_t)BLOCK_SIZE);
      mLoopInc = strideNumBlocksAlongRow;
    } else {
      Idx InnerProductPartitionIdx = blockIdx.z / num_heads;
      Idx NumInnerProductionPartitions = blockDim.z / num_heads;
      mLoopBeg = ceil_div<Idx>(num_A_cols, BLOCK_SIZE) *
                 InnerProductPartitionIdx / NumInnerProductionPartitions;
      mLoopEnd = ceil_div<Idx>(num_A_cols, BLOCK_SIZE) *
                 (InnerProductPartitionIdx + 1) / NumInnerProductionPartitions;
      mLoopInc = 1;
    }
    for (Idx m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
      // Shared memory used to store Asub and Bsub respectively
      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

      // Get sub-matrix Bsub of B
      // float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockFeat * BLOCK_SIZE];
      // float* Asub;
      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix
      if constexpr (OuterProductFlag) {
        // Get sub-matrix Asub of A
        // Asub = &A[m * BLOCK_SIZE * num_A_cols + blockRow * BLOCK_SIZE];
        As[thIdxFeat][thIdxRow] =
            (thIdxRow + (m + blockRowJobEntryBeg) * BLOCK_SIZE < numARows &&
             blockRow * BLOCK_SIZE + thIdxFeat < num_A_cols)
                ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                     AdvancedGatherAFlag>(
                      A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                      unique_srcs_and_dests_node_indices, idx_relation,
                      thIdxRow + (m + blockRowJobEntryBeg) * BLOCK_SIZE,
                      /*idx_head*/ 0, thIdxFeat + blockRow * BLOCK_SIZE,
                      /*num_heads*/ 1, num_A_cols)
                : 0.0f;

        Bs[thIdxRow][thIdxFeat] =
            ((m + blockRowJobEntryBeg) * BLOCK_SIZE + thIdxRow < numARows &&
             blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols &&
             idx_head < num_heads)
                ? GetRowMajorElement<Idx, IdxPtr, GatherBFlag,
                                     AdvancedGatherBFlag>(
                      B, B_gather_list, unique_srcs_and_dests_rel_ptr,
                      unique_srcs_and_dests_node_indices, idx_relation,
                      (m + blockRowJobEntryBeg) * BLOCK_SIZE + thIdxRow,
                      idx_head, blockFeat * BLOCK_SIZE + thIdxFeat, num_heads,
                      num_B_cols)
                : 0.0f;
      } else {
        // Get sub-matrix Asub of A
        // Asub = &A[blockRow * BLOCK_SIZE * num_A_cols + m * BLOCK_SIZE];
        As[thIdxRow][thIdxFeat] =
            (thIdxRow + blockRow * BLOCK_SIZE < numARows &&
             m * BLOCK_SIZE + thIdxFeat < num_A_cols && idx_head < num_heads)
                ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                     AdvancedGatherAFlag>(
                      A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                      unique_srcs_and_dests_node_indices, idx_relation,
                      thIdxRow + blockRow * BLOCK_SIZE + blockRowJobEntryBeg,
                      idx_head, thIdxFeat + m * BLOCK_SIZE, num_heads,
                      num_A_cols)
                : 0.0f;
        if constexpr (BWeightInsteadOfFeatureFlag) {
          // B matrix the most major dimension is num_heads, i.e., [num_heads,
          // num_B_rows_feat, num_B_cols_feat] instead of [num_nodes|edges,
          // num_heads, num_feats]
          // NB: B's num_B_cols_feat is the same as input_dim whereas
          // num_B_rows_feat is per head i.e., B dimension is [num_heads,
          // input_dim (num_A_cols), output_dim//num_heads (num_B_cols)] in
          // forward propagation or [num_heads, output_dim//num_heads
          // (num_A_cols), input_dim (num_B_cols)] in backward propagation
          Bs[thIdxRow][thIdxFeat] =
              (m * BLOCK_SIZE + thIdxRow < num_A_cols &&
               blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols &&
               idx_head < num_heads)
                  ? B[idx_head * num_A_cols * num_B_cols +
                      (m * BLOCK_SIZE + thIdxRow) * num_B_cols +
                      (blockFeat * BLOCK_SIZE + thIdxFeat)]
                  : 0.0f;
        } else {
          Bs[thIdxRow][thIdxFeat] =
              (m * BLOCK_SIZE + thIdxRow < num_A_cols &&
               blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols &&
               idx_head < num_heads)
                  ? GetRowMajorElement<Idx, IdxPtr, GatherBFlag,
                                       AdvancedGatherBFlag>(
                        B, B_gather_list, unique_srcs_and_dests_rel_ptr,
                        unique_srcs_and_dests_node_indices, idx_relation,
                        m * BLOCK_SIZE + thIdxRow, idx_head,
                        blockFeat * BLOCK_SIZE + thIdxFeat, num_heads,
                        num_B_cols)
                  : 0.0f;
        }
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
      }

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
    if constexpr (OuterProductFlag) {
      // C is weight instead of feature.
      bool WriteCInRangeFlag =
          blockRow * BLOCK_SIZE + thIdxRow < num_A_cols &&
          blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols &&
          idx_head < num_heads;
      if constexpr (!AtomicUpdateFlag) {
        CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(
            OuterProductFlag && AtomicUpdateFlag,
            "OuterproductFlag==true case must use atomic update");
      }
      if (WriteCInRangeFlag)
        atomicAdd(&C[idx_head * num_A_cols /*A is transposed in the fly*/ *
                         num_B_cols +
                     (blockRow * BLOCK_SIZE + thIdxRow) * num_B_cols +
                     blockFeat * BLOCK_SIZE + thIdxFeat],
                  Cvalue);
    } else {  //  !OuterProductFlag

      bool WriteCInRangeFlag = thIdxRow + blockRow * BLOCK_SIZE < numARows &&
                               idx_head < num_heads &&
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
        if constexpr (AtomicUpdateFlag) {
          atomicAdd(&GetRowMajorElement<Idx, IdxPtr, ScatterCFlag,
                                        AdvancedScatterCFlag>(
                        C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
                        unique_srcs_and_dests_node_indices, idx_relation,
                        thIdxRow + blockRow * BLOCK_SIZE + blockRowJobEntryBeg,
                        idx_head, blockFeat * BLOCK_SIZE + thIdxFeat, num_heads,
                        num_B_cols),
                    Cvalue);
        } else {  // !AtomicUpdateFlag
          GetRowMajorElement<Idx, IdxPtr, ScatterCFlag, AdvancedScatterCFlag>(
              C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
              unique_srcs_and_dests_node_indices, idx_relation,
              thIdxRow + blockRow * BLOCK_SIZE + blockRowJobEntryBeg, idx_head,
              blockFeat * BLOCK_SIZE + thIdxFeat, num_heads, num_B_cols) =
              Cvalue;
        }
      }
    }
  }
}

// C = A * B all are row major

// configurations of forward propagation procedures

// feat_per_edge = input * weight. C needs eid scatter list, A needs col|row_idx
// gather list. GatherA = true, ScatterC = true

// feat_compact = input * weight. C needs (col|row_idx, idx_relation, unique) ->
// offset map. A needs col|row_idx gather list, GatherA =true, ScatterC = true,
// AdvancedScatterC = true.

// configurations of backward propagation procedures

// delta node_input = delta_feat_per_edge * weightT: A needs gather list eid, C
// needs col|row_idx as scatter list. GatherA = true, ScatterC = true

// delta_weight = feat_input T * delta feat_per_edge: B needs eid as gather
// list, A needs col|row_idx as gather list. GatherA = true, GatherB = true,
// AInFlyTranspose = true

// delta input = delta feat_compact * weight T : A needs (col|row_idx,
// idx_relation, unique) -> offset map, GatherA=true, AdvancedGatherA=true

// delta weight =  feat_input T * delta feat_compact: B needs (col|row_idx,
// idx_relation, unique) -> offset map, A needs col|row_idx as gather list.
// GatherA=true, GatherB=true, AdvancedGatherB=true, AInFlyTranspose = true

// blockIdx.y == ceil_div (num_edges, BLOCK_SIZE)
template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatPerEdgeFWProp(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr A_col_row_idx_gather_list, IdxPtr A_rel_ptr,
    IdxPtr C_eid_scatter_list, Idx input_dim, Idx output_per_head_dim,
    Idx num_heads, int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, false,
                      false, Idx, IdxPtr>(
      node_feat_input, &weight[idx_relation * input_dim * output_per_head_dim],
      node_feat_per_edge, A_col_row_idx_gather_list, nullptr,
      C_eid_scatter_list, nullptr, nullptr, idx_relation,
      A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatPerEdgeFWPropACGatherScatterListIdentical(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr A_rel_ptr, IdxPtr AC_eid_gather_scatter_list, Idx input_dim,
    Idx output_per_head_dim, Idx num_heads, int* accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, false,
                      false, Idx, IdxPtr>(
      node_feat_input, &weight[idx_relation * input_dim * output_per_head_dim],
      node_feat_per_edge, AC_eid_gather_scatter_list, nullptr,
      AC_eid_gather_scatter_list, nullptr, nullptr, idx_relation,
      A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatCompactFWProp(float* node_feat_input, float* weight,
                                      float* node_feat_per_edge,
                                      IdxPtr unique_srcs_and_dests_rel_ptr,
                                      IdxPtr unique_srcs_and_dests_node_indices,
                                      Idx input_dim, Idx output_per_head_dim,
                                      Idx num_heads,
                                      int* accum_num_blocks_per_relation,
                                      Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, true,
                      false, Idx, IdxPtr>(
      node_feat_input, &weight[idx_relation * input_dim * output_per_head_dim],
      node_feat_per_edge, unique_srcs_and_dests_node_indices, nullptr,
      unique_srcs_and_dests_node_indices, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation,
      unique_srcs_and_dests_rel_ptr[idx_relation + 1] -
          unique_srcs_and_dests_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      unique_srcs_and_dests_rel_ptr[idx_relation], input_dim,
      output_per_head_dim, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputBWProp(
    float* delta_feat_per_edge, float* weight_transposed,
    float* delta_node_input, IdxPtr A_eid_gather_list, IdxPtr A_rel_ptr,
    IdxPtr C_col_row_idx_scatter_list, Idx delta_output_per_head_dim,
    Idx delta_input_dim, Idx num_heads, int* accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, false,
                      true, Idx, IdxPtr>(
      delta_feat_per_edge,
      &weight_transposed[idx_relation * delta_input_dim *
                         delta_output_per_head_dim],
      delta_node_input, A_eid_gather_list, nullptr, C_col_row_idx_scatter_list,
      nullptr, nullptr, idx_relation,
      A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], delta_output_per_head_dim, delta_input_dim,
      num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightBWProp(
    float* node_feat_input, float* delta_feat_per_edge, float* delta_weight,
    IdxPtr A_col_row_idx_gather_list, IdxPtr A_rel_ptr,
    IdxPtr B_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, true, true, false, true, false, false, false,
                      true, Idx, IdxPtr>(
      node_feat_input, delta_feat_per_edge,
      &delta_weight[idx_relation * B_delta_output_per_head_dim *
                    A_delta_input_dim],
      A_col_row_idx_gather_list, B_eid_gather_list, nullptr, nullptr, nullptr,
      idx_relation, A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], A_delta_input_dim, B_delta_output_per_head_dim,
      num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightBWPropACGatherScatterListIdentical(
    float* node_feat_input, float* delta_feat_per_edge, float* delta_weight,
    IdxPtr A_rel_ptr, IdxPtr AB_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, true, true, false, true, false, false, false,
                      true, Idx, IdxPtr>(
      node_feat_input, delta_feat_per_edge,
      &delta_weight[idx_relation * B_delta_output_per_head_dim *
                    A_delta_input_dim],
      AB_eid_gather_list, AB_eid_gather_list, nullptr, nullptr, nullptr,
      idx_relation, A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], A_delta_input_dim, B_delta_output_per_head_dim,
      num_heads);
}

// blockDim.y == ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightCompactBWProp(
    float* delta_weight, float* feat_input, float* delta_feat_compact,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_edges,
    Idx A_delta_input_dim, Idx B_delta_output_per_head_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, true, true, false, true, true, false, false,
                      true, Idx, IdxPtr>(
      feat_input, delta_feat_compact,
      &delta_weight[idx_relation * B_delta_output_per_head_dim *
                    A_delta_input_dim],
      unique_srcs_and_dests_node_indices, unique_srcs_and_dests_node_indices,
      nullptr, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation,
      unique_srcs_and_dests_rel_ptr[idx_relation + 1] -
          unique_srcs_and_dests_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      unique_srcs_and_dests_rel_ptr[idx_relation], A_delta_input_dim,
      B_delta_output_per_head_dim, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical(
    float* delta_feat_per_edge, float* weight_transposed,
    float* delta_node_input, IdxPtr A_C_eid_gather_scatter_list,
    IdxPtr A_rel_ptr, Idx delta_output_per_head_dim, Idx delta_input_dim,
    Idx num_heads, int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, false,
                      true, Idx, IdxPtr>(
      delta_feat_per_edge,
      &weight_transposed[idx_relation * delta_input_dim *
                         delta_output_per_head_dim],
      delta_node_input, A_C_eid_gather_scatter_list, nullptr,
      A_C_eid_gather_scatter_list, nullptr, nullptr, idx_relation,
      A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      A_rel_ptr[idx_relation], delta_output_per_head_dim, delta_input_dim,
      num_heads);
}

// blockDim.y == ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputCompactBWProp(
    float* delta_feat_compact, float* weight_transpose,
    float* delta_node_feat_input, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_edges,
    Idx delta_output_per_head_dim, Idx delta_input_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, true, false, false, false, false,
                      true, Idx, IdxPtr>(
      delta_feat_compact,
      &weight_transpose[idx_relation * delta_output_per_head_dim *
                        delta_input_dim],
      delta_node_feat_input, unique_srcs_and_dests_node_indices, nullptr,
      nullptr, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation,
      unique_srcs_and_dests_rel_ptr[idx_relation + 1] -
          unique_srcs_and_dests_rel_ptr[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      unique_srcs_and_dests_rel_ptr[idx_relation], delta_output_per_head_dim,
      delta_input_dim, num_heads);
}

// FIXME: separate_coo_relptrs and separate_coo_node_indices are unused in the
// following functions blockDim.y == ceil_div(A_col_row_idx_gather_list.size(),
// BLOCK_SIZE)
template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputCompactBWPropSingleSided(
    float* delta_feat_compact, float* weight_transpose,
    float* delta_node_feat_input, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, IdxPtr separate_coo_relptrs,
    IdxPtr separate_coo_node_indices, Idx num_edges,
    Idx delta_output_per_head_dim, Idx delta_input_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, true, false, false, false, false,
                      true, Idx, IdxPtr>(
      delta_feat_compact,
      &weight_transpose[idx_relation * delta_output_per_head_dim *
                        delta_input_dim],
      delta_node_feat_input, separate_coo_node_indices, nullptr, nullptr,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      idx_relation,
      separate_coo_relptrs[idx_relation + 1] -
          separate_coo_relptrs[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      separate_coo_relptrs[idx_relation], delta_output_per_head_dim,
      delta_input_dim, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatCompactFWPropSingleSided(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, IdxPtr separate_coo_relptrs,
    IdxPtr separate_coo_node_indices, Idx input_dim, Idx output_per_head_dim,
    Idx num_heads, int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, false, true, false, false, false, true, true,
                      false, Idx, IdxPtr>(
      node_feat_input, &weight[idx_relation * input_dim * output_per_head_dim],
      node_feat_per_edge, separate_coo_node_indices, nullptr,
      separate_coo_node_indices, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation,
      separate_coo_relptrs[idx_relation + 1] -
          separate_coo_relptrs[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      separate_coo_relptrs[idx_relation], input_dim, output_per_head_dim,
      num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightCompactBWPropSingleSided(
    float* delta_weight, float* feat_input, float* delta_feat_compact,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, IdxPtr separate_coo_relptrs,
    IdxPtr separate_coo_node_indices, Idx num_edges, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, Idx num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BLOCK_SIZE, true, true, false, true, true, false, false,
                      true, Idx, IdxPtr>(
      feat_input, delta_feat_compact,
      &delta_weight[idx_relation * B_delta_output_per_head_dim *
                    A_delta_input_dim],
      separate_coo_node_indices, separate_coo_node_indices, nullptr,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      idx_relation,
      separate_coo_relptrs[idx_relation + 1] -
          separate_coo_relptrs[idx_relation],
      accum_num_blocks_per_relation[idx_relation],
      (accum_num_blocks_per_relation[idx_relation + 1] -
       accum_num_blocks_per_relation[idx_relation]),
      separate_coo_relptrs[idx_relation], A_delta_input_dim,
      B_delta_output_per_head_dim, num_heads);
}
