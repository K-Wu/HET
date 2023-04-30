#pragma once
#include <cuda_runtime.h>
// #include "cuda.h"
#include "my_shmem_sgemm_func_functor.cu.h"
#include "utils.cu.h"

// vanilla tiled shmem gemm code from
// http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
//@@ Example of grid and block configuration
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// dim3 dimGrid(B.width / dimBlock.x, A.height /
// dimBlock.y, num_heads * num_partitions_along_Acol_Brow );
// MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
// C = A * B (all are row-major matrices)
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, bool OuterProductFlag, bool GatherAFlag,
          bool AdvancedGatherAFlag, bool GatherBFlag, bool AdvancedGatherBFlag,
          bool ScatterCFlag, bool AdvancedScatterCFlag, bool AtomicUpdateFlag,
          typename Idx, typename IdxPtr, bool A_num_head_one_flag,
          bool B_num_head_one_flag, bool C_num_head_one_flag>
class _basic_MatMulKernel<
    false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, SHMEM_BLOCK_SIZE,
    OuterProductFlag, GatherAFlag, AdvancedGatherAFlag, GatherBFlag,
    AdvancedGatherBFlag, ScatterCFlag, AdvancedScatterCFlag, AtomicUpdateFlag,
    Idx, IdxPtr, A_num_head_one_flag, B_num_head_one_flag,
    C_num_head_one_flag> {
 public:
  __device__ __forceinline__ static void execute_function(
      float* A, float* B, float* C, IdxPtr A_gather_list, IdxPtr B_gather_list,
      IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
      IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx numARows,
      Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    // num_B_cols is output_dim//num_heads as forward propagation weight,
    // output_dim//num_heads as backward propagation weight, and in_feat_dim as
    // features or delta features. num_A_cols is input_dim as forward
    // propagation input feature, output_dim//num_heads as delta input feature
    constexpr bool BWeightInsteadOfFeatureFlag = !OuterProductFlag;

    // NB: when OuterProductFlag is true and the model is RGAT, num_heads of the
    // input features is always 1 and the other is num_heads. In case of RGCN,
    // both are 1. In case of HGT, both are num_heads except for the kqva linear
    // (both are 1). In the implementation, when OuterProductFlag is true, A is
    // always input and B the gradient output feature. It is safe to pass
    // num_A_cols as in_dim.
    constexpr int THREADING_BLOCK_SIZE_X =
        COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE;
    constexpr int THREADING_BLOCK_SIZE_Y =
        COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE;
    constexpr int COARSEN_DIVISOR_FACTOR =
        (COARSEN_FACTOR_2_FLAG_X ? 2 : 1) * (COARSEN_FACTOR_2_FLAG_Y ? 2 : 1);
    // Block row and column
    int idx_head = blockIdx.z % num_heads;

    if constexpr (OuterProductFlag) {
      CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(OuterProductFlag, AtomicUpdateFlag,
                                          "");
    } else {
      assert((blockDim.z == num_heads));
    }

    // Idx blockRow = blockIdx.y - blockIdxAlongRowBeg;
    int blockFeat = blockIdx.x;  // when OuterProductFlag==True, it is in [0,
                                 // output_dim//num_heads)

    int blockRowLoopBeg, blockRowLoopEnd, blockRowLoopInc;
    if constexpr (OuterProductFlag) {
      blockRowLoopBeg =
          blockIdx.y;  // [0, input_dim) // NB: When OuterProductFlag is true,
                       // the delta_weight height/width is assigned to
                       // theblockIdx.y-dimension, and blockIdxAssignment is
                       // assigned to m loop instead. Therefore, the
                       // blockIdxAlongRowBeg bias is applied to the m loop too.
      blockRowLoopEnd = blockIdx.y + 1;
      blockRowLoopInc = 1;
    } else {
      blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
      blockRowLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE);
      blockRowLoopInc = strideNumBlocksAlongRow;
    }

    for (int blockRow = blockRowLoopBeg; blockRow < blockRowLoopEnd;
         blockRow += blockRowLoopInc) {
      // NB: blockTask == blockIdx.x / ceil_div( num_B_cols, BLOCK_SIZE)

      // Each thread block computes one sub-matrix Csub of C
      // float* Csub = &C[blockRow * BLOCK_SIZE * num_B_cols + blockFeat *
      // BLOCK_SIZE]; Each thread computes one element of Csub by accumulating
      // results into Cvalue
      float Cvalue[COARSEN_DIVISOR_FACTOR] = {};
      // Thread row and column within Csub
      int thIdxRow_initial = threadIdx.y;
      int thIdxFeat_initial = threadIdx.x;
      if constexpr (COARSEN_FACTOR_2_FLAG_X || COARSEN_FACTOR_2_FLAG_Y) {
        // redo the thread indexing
        int thIdx = threadIdx.y * THREADING_BLOCK_SIZE_X + threadIdx.x;
        thIdxRow_initial = thIdx / SHMEM_BLOCK_SIZE;
        thIdxFeat_initial = thIdx % SHMEM_BLOCK_SIZE;
      }
      // Loop over all the sub-matrices of A and B that are
      // required to compute Csub
      // Multiply each pair of sub-matrices together
      // and accumulate the results

      int mLoopBeg, mLoopEnd, mLoopInc;
      if constexpr (OuterProductFlag) {
        // the block configuration scheme is different when OuterProductFlag
        // compared with !OuterProductFlag case when OuterProductFlag, the block
        // configuration is (num_output_per_head_dim//BLOCK_SIZE,
        // num_input_dim//BLOCK_SIZE,num_heads * num_edges) when
        // !OuterProductFlag, the block configuration is
        // (num_output_per_head_dim//BLOCK_SIZE,num_edges,num_heads)
        int blockAssignmentIdx = blockIdx.z / num_heads;
        mLoopBeg = blockAssignmentIdx - blockIdxAlongRowBeg;
        mLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE);
        mLoopInc = strideNumBlocksAlongRow;
      } else {
        int InnerProductPartitionIdx = blockIdx.z / num_heads;
        int NumInnerProductionPartitions = blockDim.z / num_heads;
        mLoopBeg = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE) *
                   InnerProductPartitionIdx / NumInnerProductionPartitions;
        mLoopEnd = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE) *
                   (InnerProductPartitionIdx + 1) /
                   NumInnerProductionPartitions;
        mLoopInc = 1;
      }
      for (int m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];
        __shared__ float Bs[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];

        // Get sub-matrix Bsub of B
        // float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockFeat *
        // BLOCK_SIZE]; float* Asub; Load Asub and Bsub from device memory to
        // shared memory Each thread loads one element of each sub-matrix
        for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR;
             loadLoopIdx++) {
          int thIdxRow =
              thIdxRow_initial +
              loadLoopIdx * (SHMEM_BLOCK_SIZE / COARSEN_DIVISOR_FACTOR);
          int thIdxFeat = thIdxFeat_initial;
          if constexpr (OuterProductFlag) {
            // Get sub-matrix Asub of A
            // Asub = &A[m * BLOCK_SIZE * num_A_cols + blockRow * BLOCK_SIZE];
            As[thIdxFeat][thIdxRow] =
                (thIdxRow + (m)*SHMEM_BLOCK_SIZE <  //+ blockRowJobEntryBeg <
                     numARows &&
                 blockRow * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols)
                    ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                         AdvancedGatherAFlag>(
                          A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                          unique_srcs_and_dests_node_indices, idx_relation,
                          thIdxRow + (m)*SHMEM_BLOCK_SIZE + blockRowJobEntryBeg,
                          /*idx_head*/ 0,
                          thIdxFeat + blockRow * SHMEM_BLOCK_SIZE,
                          /*num_heads*/ 1, num_A_cols)
                    : 0.0f;

            Bs[thIdxRow][thIdxFeat] =
                ((m)*SHMEM_BLOCK_SIZE + thIdxRow <  //+ blockRowJobEntryBeg <
                     numARows &&
                 blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols &&
                 idx_head < num_heads)
                    ? GetRowMajorElement<Idx, IdxPtr, GatherBFlag,
                                         AdvancedGatherBFlag>(
                          B, B_gather_list, unique_srcs_and_dests_rel_ptr,
                          unique_srcs_and_dests_node_indices, idx_relation,
                          (m)*SHMEM_BLOCK_SIZE + blockRowJobEntryBeg + thIdxRow,
                          B_num_head_one_flag ? 0 : idx_head,
                          blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat,
                          B_num_head_one_flag ? 1 : num_heads, num_B_cols)
                    : 0.0f;
          } else {
            // Get sub-matrix Asub of A
            // Asub = &A[blockRow * BLOCK_SIZE * num_A_cols + m * BLOCK_SIZE];
            As[thIdxRow][thIdxFeat] =
                (thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
                 m * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols &&
                 idx_head < num_heads)
                    ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                         AdvancedGatherAFlag>(
                          A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                          unique_srcs_and_dests_node_indices, idx_relation,
                          thIdxRow + blockRow * SHMEM_BLOCK_SIZE +
                              blockRowJobEntryBeg,
                          A_num_head_one_flag ? 0 : idx_head,
                          thIdxFeat + m * SHMEM_BLOCK_SIZE,
                          A_num_head_one_flag ? 1 : num_heads, num_A_cols)
                    : 0.0f;
            if constexpr (BWeightInsteadOfFeatureFlag) {
              // B matrix the most major dimension is num_heads, i.e.,
              // [num_heads, num_B_rows_feat, num_B_cols_feat] instead of
              // [num_nodes|edges, num_heads, num_feats] NB: B's num_B_cols_feat
              // is the same as input_dim whereas num_B_rows_feat is per head
              // i.e., B dimension is [num_heads, input_dim (num_A_cols),
              // output_dim//num_heads (num_B_cols)] in forward propagation or
              // [num_heads, output_dim//num_heads (num_A_cols), input_dim
              // (num_B_cols)] in backward propagation NB: incorporate
              // num_head_one_flag
              Bs[thIdxRow][thIdxFeat] =
                  (m * SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols &&
                   blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols &&
                   idx_head < num_heads)
                      ? B[(B_num_head_one_flag ? 0 : idx_head) * num_A_cols *
                              num_B_cols +
                          (m * SHMEM_BLOCK_SIZE + thIdxRow) * num_B_cols +
                          (blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat)]
                      : 0.0f;
            } else {
              Bs[thIdxRow][thIdxFeat] =
                  (m * SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols &&
                   blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols &&
                   idx_head < num_heads)
                      ? GetRowMajorElement<Idx, IdxPtr, GatherBFlag,
                                           AdvancedGatherBFlag>(
                            B, B_gather_list, unique_srcs_and_dests_rel_ptr,
                            unique_srcs_and_dests_node_indices, idx_relation,
                            m * SHMEM_BLOCK_SIZE + thIdxRow,
                            B_num_head_one_flag ? 0 : idx_head,
                            blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat,
                            B_num_head_one_flag ? 1 : num_heads, num_B_cols)
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
            //                  blockFeat * BLOCK_SIZE + thIdxFeat < num_B_cols
            //                  && idx_head < num_heads;
            //     bool aflag = thIdxRow + blockRow * BLOCK_SIZE < numARows &&
            //                  m * BLOCK_SIZE + thIdxFeat < num_A_cols &&
            //                  idx_head < num_heads;

            //     printf(
            //         "0 found(WriteCInRangeFlag)!!! (thIdxRow %ld, blockRow
            //         %ld, " "blockRowJobEntryBeg "
            //         "%ld, numARows %ld), (thIdxFeat %ld, blockFeat %ld,
            //         num_B_cols "
            //         "%ld), (idx_head %ld, num_head %ld) (idx_relation %ld) "
            //         "(mLoopBeg %ld mLoopEnd %ld m%ld) (bweightflag %d aflag
            //         %d " "bflag %d) (shmem A %f B %f)\n", thIdxRow, blockRow,
            //         blockRowJobEntryBeg, numARows, thIdxFeat, blockFeat,
            //         num_B_cols, idx_head, num_heads, idx_relation, mLoopBeg,
            //         mLoopEnd, m, BWeightInsteadOfFeatureFlag, aflag, bflag,
            //         As[thIdxRow][thIdxFeat], Bs[thIdxRow][thIdxFeat]);
            //   }
            // }
          }
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < SHMEM_BLOCK_SIZE; ++e) {
          for (int idx_coarsen_factor = 0;
               idx_coarsen_factor < COARSEN_DIVISOR_FACTOR;
               idx_coarsen_factor++) {
            Cvalue[idx_coarsen_factor] +=
                As[thIdxRow_initial +
                   idx_coarsen_factor *
                       (SHMEM_BLOCK_SIZE / COARSEN_DIVISOR_FACTOR)][e] *
                Bs[e][thIdxFeat_initial];
          }
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }

      // Write Csub to device memory
      // Each thread writes one element
      for (int storeLoopIdx = 0; storeLoopIdx < COARSEN_DIVISOR_FACTOR;
           storeLoopIdx++) {
        int thIdxRow =
            thIdxRow_initial +
            storeLoopIdx * (SHMEM_BLOCK_SIZE / COARSEN_DIVISOR_FACTOR);
        int thIdxFeat = thIdxFeat_initial;
        if constexpr (OuterProductFlag) {
          // C is weight instead of feature.
          bool WriteCInRangeFlag =
              blockRow * SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols &&
              blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols &&
              idx_head < num_heads;
          if constexpr (!AtomicUpdateFlag) {
            CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(
                OuterProductFlag && AtomicUpdateFlag,
                "OuterproductFlag==true case must use atomic update");
          }
          if (WriteCInRangeFlag) {
            // NB: offset dependent on whether one-side num_head is 1
            atomicAdd(
                &C[(C_num_head_one_flag ? 0 : idx_head) *
                       num_A_cols /*A is transposed in the fly*/ * num_B_cols +
                   (blockRow * SHMEM_BLOCK_SIZE + thIdxRow) * num_B_cols +
                   blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat],
                Cvalue[storeLoopIdx]);
          }
        } else {  //  !OuterProductFlag

          bool WriteCInRangeFlag =
              thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
              idx_head < num_heads &&
              blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols;
          // if (ScatterCFlag && !AdvancedScatterCFlag) {
          //   if (C_scatter_list[thIdxRow + blockRow * BLOCK_SIZE +
          //                      blockRowJobEntryBeg] == 0) {
          //     printf(
          //         "0 found(%d)!!! (thIdxRow %ld, blockRow %ld,
          //         blockRowJobEntryBeg "
          //         "%ld, numARows %ld), (thIdxFeat %ld, blockFeat %ld,
          //         num_B_cols
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
              // print GetRowMajorElement arguments
              // if constexpr (!OuterProductFlag && !GatherAFlag &&
              // !AdvancedGatherAFlag && !GatherBFlag && !AdvancedGatherBFlag &&
              // ScatterCFlag && !AdvancedScatterCFlag && AtomicUpdateFlag)
              // if (ScatterCFlag && !AdvancedScatterCFlag)
              // printf("C %p C_scatter_list %p unique_srcs_and_dests_rel_ptr %p
              // "
              //        "unique_srcs_and_dests_node_indices %p idx_relation %ld
              //        " "idxRow %ld scatter_list[idxRow] %ld idxFeat %ld
              //        idx_head %ld num_heads %ld " "num_B_cols %ld, numARows
              //        %ld\n", C, C_scatter_list,
              //        unique_srcs_and_dests_rel_ptr,
              //        unique_srcs_and_dests_node_indices, idx_relation,
              //        thIdxRow + blockRow * BLOCK_SIZE,
              //        C_scatter_list[thIdxRow
              //        + blockRow * BLOCK_SIZE], blockFeat * BLOCK_SIZE +
              //        thIdxFeat, idx_head, num_heads, num_B_cols, numARows);
              atomicAdd(&GetRowMajorElement<Idx, IdxPtr, ScatterCFlag,
                                            AdvancedScatterCFlag>(
                            C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
                            unique_srcs_and_dests_node_indices, idx_relation,
                            thIdxRow + blockRow * SHMEM_BLOCK_SIZE +
                                blockRowJobEntryBeg,
                            C_num_head_one_flag ? 0 : idx_head,
                            blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat,
                            C_num_head_one_flag ? 1 : num_heads, num_B_cols),
                        Cvalue[storeLoopIdx]);
            } else {  // !AtomicUpdateFlag
              GetRowMajorElement<Idx, IdxPtr, ScatterCFlag,
                                 AdvancedScatterCFlag>(
                  C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices, idx_relation,
                  thIdxRow + blockRow * SHMEM_BLOCK_SIZE + blockRowJobEntryBeg,
                  C_num_head_one_flag ? 0 : idx_head,
                  blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat,
                  C_num_head_one_flag ? 1 : num_heads, num_B_cols) =
                  Cvalue[storeLoopIdx];
            }
          }
        }
      }
    }
  }
};
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
template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool A_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void __launch_bounds__(256, 3)
    HET_RGNNFeatPerEdgeFWProp(float* node_feat_input, float* weight,
                              float* node_feat_per_edge,
                              IdxPtr A_col_row_idx_gather_list,
                              IdxPtr A_rel_ptr, IdxPtr C_eid_scatter_list,
                              Idx input_dim, Idx output_per_head_dim,
                              int num_heads, int* accum_num_blocks_per_relation,
                              Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      SHMEM_BLOCK_SIZE, false, true, false, false, false, true,
                      false, false, Idx, IdxPtr, A_num_head_one_flag, false,
                      false>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (A_num_head_one_flag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
          node_feat_per_edge, A_col_row_idx_gather_list, nullptr,
          C_eid_scatter_list, nullptr, nullptr, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

// for HGT nodewise linear layers
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HET_RGNNMatmulNoScatterGatherListFwOrBwProp(
    float* node_feat_input, float* weights, float* linear_projected_node_feat,
    IdxPtr ntype_ptrs, int* accum_num_blocks_per_ntype, Idx num_ntypes,
    Idx input_dim, Idx output_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_ntype = binary_search<int, int*>(
      num_ntypes, accum_num_blocks_per_ntype, idx_block_assignment);
  _basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      false, false, false, false, false, false, false, false, Idx, IdxPtr, true,
      true, true>::execute_function(node_feat_input, weights,
                                    linear_projected_node_feat, nullptr,
                                    nullptr, nullptr, nullptr, nullptr, 0,
                                    ntype_ptrs[idx_ntype + 1] -
                                        ntype_ptrs[idx_ntype],
                                    accum_num_blocks_per_ntype[idx_ntype],
                                    (accum_num_blocks_per_ntype[idx_ntype + 1] -
                                     accum_num_blocks_per_ntype[idx_ntype]),
                                    ntype_ptrs[idx_ntype], input_dim,
                                    output_dim, 1);
}

// for HGT nodewise linear layers
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HET_RGNNDeltaWeightNoScatterGatherListBWProp(
    float* node_feat_input, float* delta_feat, float* delta_weight,
    IdxPtr ntype_ptrs, Idx A_input_dim, Idx B_delta_output_dim,
    int* accum_num_blocks_per_ntype, Idx num_ntypes) {
  Idx idx_block_assignment = blockIdx.z;
  Idx idx_ntype = binary_search<int, int*>(
      num_ntypes, accum_num_blocks_per_ntype, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, true, false, false, false, false, false,
                      false, true, Idx, IdxPtr, true, true, true>::
      execute_function(
          node_feat_input, delta_feat,
          &delta_weight[idx_ntype * A_input_dim * B_delta_output_dim], nullptr,
          nullptr, nullptr, nullptr, nullptr, idx_ntype,
          ntype_ptrs[idx_ntype + 1] - ntype_ptrs[idx_ntype],
          accum_num_blocks_per_ntype[idx_ntype],
          (accum_num_blocks_per_ntype[idx_ntype + 1] -
           accum_num_blocks_per_ntype[idx_ntype]),
          ntype_ptrs[idx_ntype], A_input_dim, B_delta_output_dim, 1);
}

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool A_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNFeatPerEdgeFWPropACGatherScatterListIdentical(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr A_rel_ptr, IdxPtr AC_eid_gather_scatter_list, Idx input_dim,
    Idx output_per_head_dim, int num_heads, int* accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, false, true, false, false, false, true,
                      false, false, Idx, IdxPtr, A_num_head_one_flag, false,
                      false>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (A_num_head_one_flag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
          node_feat_per_edge, AC_eid_gather_scatter_list, nullptr,
          AC_eid_gather_scatter_list, nullptr, nullptr, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool A_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void __launch_bounds__(256, 3)
    HET_RGNNFeatCompactFWProp(float* node_feat_input, float* weight,
                              float* node_feat_per_edge,
                              IdxPtr unique_srcs_and_dests_rel_ptr,
                              IdxPtr unique_srcs_and_dests_node_indices,
                              Idx input_dim, Idx output_per_head_dim,
                              int num_heads, int* accum_num_blocks_per_relation,
                              Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, WORK_BLOCK_SIZE,
      false, true, false, false, false, false,  // NB: no need to scatter C
      false, false, Idx, IdxPtr, A_num_head_one_flag, false, false>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (A_num_head_one_flag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
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

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool C_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaNodeFeatInputBWProp(
    float* delta_feat_per_edge, float* weight_transposed,
    float* delta_node_input, IdxPtr A_eid_gather_list, IdxPtr A_rel_ptr,
    IdxPtr C_col_row_idx_scatter_list, Idx delta_output_per_head_dim,
    Idx delta_input_dim, int num_heads, int* accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, false, true, false, false, false, true,
                      false, true, Idx, IdxPtr, false, false,
                      C_num_head_one_flag>::
      execute_function(
          delta_feat_per_edge,
          &weight_transposed[idx_relation *
                             (C_num_head_one_flag ? num_heads : 1) *
                             delta_input_dim * delta_output_per_head_dim],
          delta_node_input, A_eid_gather_list, nullptr,
          C_col_row_idx_scatter_list, nullptr, nullptr, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], delta_output_per_head_dim, delta_input_dim,
          num_heads);
}

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool A_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightBWProp(
    float* node_feat_input, float* delta_feat_per_edge, float* delta_weight,
    IdxPtr A_col_row_idx_gather_list, IdxPtr A_rel_ptr,
    IdxPtr B_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, int num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, true, true, false, true, false, false,
                      false, true, Idx, IdxPtr, A_num_head_one_flag, false,
                      false>::
      execute_function(
          node_feat_input, delta_feat_per_edge,
          &delta_weight[idx_relation * (A_num_head_one_flag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          A_col_row_idx_gather_list, B_eid_gather_list, nullptr, nullptr,
          nullptr, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], A_delta_input_dim,
          B_delta_output_per_head_dim, num_heads);
}

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool A_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightBWPropACGatherScatterListIdentical(
    float* node_feat_input, float* delta_feat_per_edge, float* delta_weight,
    IdxPtr A_rel_ptr, IdxPtr AB_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, int num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, true, true, false, true, false, false,
                      false, true, Idx, IdxPtr, A_num_head_one_flag, false,
                      false>::
      execute_function(
          node_feat_input, delta_feat_per_edge,
          &delta_weight[idx_relation * (A_num_head_one_flag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          AB_eid_gather_list, AB_eid_gather_list, nullptr, nullptr, nullptr,
          idx_relation, A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], A_delta_input_dim,
          B_delta_output_per_head_dim, num_heads);
}

// blockDim.y == ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool B_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightCompactBWProp(
    float* delta_feat_compact, float* feat_input, float* delta_weight,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_edges,
    Idx A_delta_input_dim, Idx B_delta_output_per_head_dim, int num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, true, true, false, false, false, false,
                      false, true, Idx, IdxPtr, false, B_num_head_one_flag,
                      false>::
      execute_function(
          feat_input, delta_feat_compact,
          &delta_weight[idx_relation * (B_num_head_one_flag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          unique_srcs_and_dests_node_indices,
          unique_srcs_and_dests_node_indices, nullptr,
          unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
          idx_relation,
          unique_srcs_and_dests_rel_ptr[idx_relation + 1] -
              unique_srcs_and_dests_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          unique_srcs_and_dests_rel_ptr[idx_relation], A_delta_input_dim,
          B_delta_output_per_head_dim, num_heads);
}

template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool C_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaNodeFeatInputBWPropACGatherScatterListIdentical(
    float* delta_feat_per_edge, float* weight_transposed,
    float* delta_node_input, IdxPtr A_C_eid_gather_scatter_list,
    IdxPtr A_rel_ptr, Idx delta_output_per_head_dim, Idx delta_input_dim,
    int num_heads, int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, false, true, false, false, false, true,
                      false, true, Idx, IdxPtr, false, false,
                      C_num_head_one_flag>::
      execute_function(
          delta_feat_per_edge,
          &weight_transposed[idx_relation *
                             (C_num_head_one_flag ? num_heads : 1) *
                             delta_input_dim * delta_output_per_head_dim],
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
template <
    bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
    int WORK_BLOCK_SIZE, typename Idx, typename IdxPtr,
    bool C_num_head_one_flag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaNodeFeatInputCompactBWProp(
    float* delta_feat_compact, float* weight_transpose,
    float* delta_node_feat_input, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_edges,
    Idx delta_output_per_head_dim, Idx delta_input_dim, int num_heads,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y,
                      WORK_BLOCK_SIZE, false, false, false, false, false, true,
                      false, true, Idx, IdxPtr, false, false,
                      C_num_head_one_flag>::
      execute_function(
          delta_feat_compact,
          &weight_transpose[idx_relation *
                            (C_num_head_one_flag ? num_heads : 1) *
                            delta_output_per_head_dim * delta_input_dim],
          delta_node_feat_input, unique_srcs_and_dests_node_indices, nullptr,
          unique_srcs_and_dests_node_indices, unique_srcs_and_dests_rel_ptr,
          unique_srcs_and_dests_node_indices, idx_relation,
          unique_srcs_and_dests_rel_ptr[idx_relation + 1] -
              unique_srcs_and_dests_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          unique_srcs_and_dests_rel_ptr[idx_relation],
          delta_output_per_head_dim, delta_input_dim, num_heads);
}

// NB: SingleEnded compact matmul is a mal-purposed API because the additional
// knowledge of separate coo node index does not help reduce the computation,
// i.e., the number of entries to be output. The current indexing scheme in the
// implementation is a mixture of compact and per-edge schemes.  Additionally,
// datasets we are dealing with are all added with reverse edges. So it is even
// meaningless to create two unique node indices list.

// NB: In the following
// functions, separate_coo_relptrs and separate_coo_node_indices are used as
// gather/scatter list and work assignment offset pointers, instead of the
// unique_srcs_and_dests pair in the above functions. blockDim.y ==
// ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
