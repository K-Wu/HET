#pragma once
#include <cuda_runtime.h>
#include "cuda.h"
#include "my_shmem_sgemm_func_rgcn_hgt_functor.cu.h"
#include "utils.cu.h"

template <bool COARSEN_FACTOR_2_FLAG, int THREADING_BLOCK_SIZE, typename Idx,
          typename IdxPtr, bool HGT_INSTEAD_OF_RGCN_FLAG, bool OuterProductFlag,
          int DoInnerProductSwitch,
          bool InnerProductGatherListNodeInsteadOfEdge, bool NoEdgeNormFlag>
class _simplified_basic_MatMulKernel<
    false, COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, Idx, IdxPtr,
    HGT_INSTEAD_OF_RGCN_FLAG, OuterProductFlag, DoInnerProductSwitch,
    InnerProductGatherListNodeInsteadOfEdge, NoEdgeNormFlag> {
 public:
  // vanilla tiled shmem gemm code from
  // http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
  //@@ Example of grid and block configuration
  //	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 dimGrid(B.width / dimBlock.x, A.height /
  // dimBlock.y, num_heads * num_partitions_along_Acol_Brow );
  // MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  // C = A * B (all are row-major matrices)
  // for each edge, grad enorm = message * grad_output = input *
  // grad_input_from_this_edge which we can compute  by the way when calculating
  // delta node feat
  // NB: this do grad norm is further generalized to be extended to do attention
  // score in the fly as both are inner product inner_product_other_term is
  // input_node_feat for grad_norm, and input_node_term for attention_score
  // calculation DoInnerProductSwitch 0: no inner product, 1: do inner product,
  // 2: do inner product and do no C inner_product is grad_edge_norm or
  // unnormalized_attn_score
  __device__ __forceinline__ static void exec_function(
      float* A, float* B, float* C, float* edge_norm, float* inner_product,
      float* input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
      IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids, Idx idx_relation,
      Idx numARows, Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, Idx num_heads) {
    // num_B_cols is output_dim//num_heads as forward propagation weight,
    // output_dim//num_heads as backward propagation weight, and in_feat_dim as
    // features or delta features. num_A_cols is input_dim as forward
    // propagation input feature, output_dim//num_heads as delta input feature

    // NB: when OuterProductFlag is true, num_heads of the input features is
    // always 1 and the other is num_heads, at least in case of RGAT. In the
    // implementation, when OuterProductFlag is true, A is always input and B
    // the gradient output feature. It is safe to pass num_A_cols as in_dim.

    // FIXME: check if we can reduce the use of norm by applying only at the end

    // Block row and column
    // if constexpr(!DoHalfGradNormFlag){
    //   assert(inner_product == nullptr);
    // }
    if constexpr (!HGT_INSTEAD_OF_RGCN_FLAG) {
      // assuming this case is RGCN and there is no multiple head
      assert((blockDim.z == 1));
    }  // otherwise assuming HGT
    constexpr int SHMEM_BLOCK_SIZE =
        COARSEN_FACTOR_2_FLAG ? THREADING_BLOCK_SIZE * 2 : THREADING_BLOCK_SIZE;
    Idx idx_head = blockIdx.z;
    IdxPtr A_gather_list;
    IdxPtr C_scatter_list;
    IdxPtr B_gather_list;
    IdxPtr inner_product_term_gather_list;
    if constexpr (OuterProductFlag) {
      // A is input feature, B is gradient output feature
      A_gather_list = separate_coo_row_idx;
      B_gather_list = separate_coo_col_idx;
    } else {
      A_gather_list = separate_coo_row_idx;
      C_scatter_list = separate_coo_col_idx;
    }
    if constexpr (DoInnerProductSwitch > 0) {
      if constexpr (InnerProductGatherListNodeInsteadOfEdge) {
        inner_product_term_gather_list = separate_coo_col_idx;
      } else {
        inner_product_term_gather_list = separate_coo_eids;
      }
    }
    // Idx blockRow = blockIdx.y - blockIdxAlongRowBeg;
    Idx blockFeat = blockIdx.x;  // when OuterProductFlag==True, it is in [0,
                                 // output_dim//num_heads)

    Idx blockRowLoopBeg, blockRowLoopEnd, blockRowLoopInc;
    if constexpr (OuterProductFlag) {
      blockRowLoopBeg =
          blockIdx.y;  // [0, input_dim) // check my_shmem_sgemm_func.cu.h NB on
                       // why -blockIdxAlongRowBeg bias is not applied here but
                       // applied to the m loop
      blockRowLoopEnd = blockIdx.y + 1;
      blockRowLoopInc = 1;
    } else {
      blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
      blockRowLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE);
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
      float Cvalue_1 = 0.0f;
      float Cvalue_2 = 0.0f;
      float Cvalue_3 = 0.0f;

      // Thread row and column within Csub
      Idx thIdxRow_initial = threadIdx.y;
      Idx thIdxFeat_initial = threadIdx.x;
      if constexpr (COARSEN_FACTOR_2_FLAG) {
        Idx thIdx = threadIdx.y * blockDim.x + threadIdx.x;
        thIdxRow_initial = thIdx / SHMEM_BLOCK_SIZE;
        thIdxFeat_initial = thIdx % SHMEM_BLOCK_SIZE;
      }
      // Loop over all the sub-matrices of A and B that are
      // required to compute Csub
      // Multiply each pair of sub-matrices together
      // and accumulate the results

      Idx mLoopBeg, mLoopEnd, mLoopInc;
      if constexpr (OuterProductFlag) {
        Idx blockAssignmentIdx = blockIdx.z / num_heads;
        mLoopBeg = blockAssignmentIdx - blockIdxAlongRowBeg;
        mLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE);
        mLoopInc = strideNumBlocksAlongRow;
      } else {
        mLoopBeg = 0;
        mLoopEnd = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE);
        mLoopInc = 1;
      }
      for (Idx m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];
        __shared__ float Bs[SHMEM_BLOCK_SIZE][SHMEM_BLOCK_SIZE];

        // Get sub-matrix Bsub of B
        // float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockFeat *
        // BLOCK_SIZE]; float* Asub; Load Asub and Bsub from device memory to
        // shared memory Each thread loads one element of each sub-matrix

        // Get sub-matrix Asub of A
        // Asub = &A[blockRow * BLOCK_SIZE * num_A_cols + m * BLOCK_SIZE];
        for (Idx loadLoopIdx = 0; loadLoopIdx < (COARSEN_FACTOR_2_FLAG ? 4 : 1);
             loadLoopIdx++) {
          Idx thIdxRow =
              thIdxRow_initial + loadLoopIdx * (SHMEM_BLOCK_SIZE / 4);
          Idx thIdxFeat = thIdxFeat_initial;
          if constexpr (OuterProductFlag) {
            As[thIdxFeat][thIdxRow] =
                (thIdxRow + (m)*SHMEM_BLOCK_SIZE + blockRowJobEntryBeg <
                     numARows &&
                 blockRow * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols)
                    ? A[A_gather_list[thIdxRow + (m)*SHMEM_BLOCK_SIZE +
                                      blockRowJobEntryBeg] *
                            num_A_cols * num_heads +
                        num_A_cols * idx_head +
                        (blockRow * SHMEM_BLOCK_SIZE + thIdxFeat)] *
                          (NoEdgeNormFlag
                               ? 1.0f
                               : edge_norm
                                     [separate_coo_eids[thIdxRow +
                                                        (m)*SHMEM_BLOCK_SIZE +
                                                        blockRowJobEntryBeg] *
                                          num_heads +
                                      idx_head])
                    : 0.0f;
            Bs[thIdxRow][thIdxFeat] =
                ((m)*SHMEM_BLOCK_SIZE + thIdxRow + blockRowJobEntryBeg <
                     numARows &&
                 blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols)
                    ? B[B_gather_list[(m)*SHMEM_BLOCK_SIZE + thIdxRow +
                                      blockRowJobEntryBeg] *
                            num_B_cols * num_heads +
                        num_B_cols * idx_head +
                        (blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat)]
                    : 0.0f;
          } else {
            // printf("blockRow %d blockFeat %d thIdxRow %d thIdxFeat %d
            // AloadFlag %d BloadFlag %d \n", blockRow, blockFeat, thIdxRow,
            // thIdxFeat, (thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
            //      m * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols), (m *
            //      SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols && blockFeat *
            //      SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols));
            As[thIdxRow][thIdxFeat] =
                (thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
                 m * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols)
                    ? A[A_gather_list[thIdxRow + blockRow * SHMEM_BLOCK_SIZE +
                                      blockRowJobEntryBeg] *
                            num_A_cols * num_heads +
                        num_A_cols * idx_head +
                        (thIdxFeat + m * SHMEM_BLOCK_SIZE)] *
                          (NoEdgeNormFlag
                               ? 1.0f
                               : edge_norm
                                     [separate_coo_eids[thIdxRow +
                                                        blockRow *
                                                            SHMEM_BLOCK_SIZE +
                                                        blockRowJobEntryBeg] *
                                          num_heads +
                                      idx_head])
                    : 0.0f;
            // if constexpr (HGT_INSTEAD_OF_RGCN_FLAG) {
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
            // NB: this indexing scheme works for both cases whether num_head is
            // 1

            Bs[thIdxRow][thIdxFeat] =
                (m * SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols &&
                 blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols)
                    ? B[(m * SHMEM_BLOCK_SIZE + thIdxRow) * num_B_cols *
                            num_heads +
                        idx_head * num_B_cols +
                        (blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat)]
                    : 0.0f;
          }
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

        __syncthreads();
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        // Multiply Asub and Bsub together
        for (int e = 0; e < SHMEM_BLOCK_SIZE; ++e) {
          Cvalue += As[thIdxRow_initial][e] * Bs[e][thIdxFeat_initial];
          if constexpr (COARSEN_FACTOR_2_FLAG) {
            Cvalue_1 += As[thIdxRow_initial + 1 * (SHMEM_BLOCK_SIZE / 4)][e] *
                        Bs[e][thIdxFeat_initial];
            Cvalue_2 += As[thIdxRow_initial + 2 * (SHMEM_BLOCK_SIZE / 4)][e] *
                        Bs[e][thIdxFeat_initial];
            Cvalue_3 += As[thIdxRow_initial + 3 * (SHMEM_BLOCK_SIZE / 4)][e] *
                        Bs[e][thIdxFeat_initial];
          }
        }

        __syncthreads();
        // if constexpr (OuterProductFlag && DoHalfGradNormFlag){
        //   // only diagonal threading block needs to do the work
        //   // only the first few warp needs to do the work
        //   // each edge (determined by separate_coo_eid[work_item_idx]) get
        //   (delta_input_from_this_edge * input where input node id is the same
        //   as the edge's src node id, i.e.,
        //   separate_coo_row_idx[work_item_idx]) if (COARSEN_FACTOR_2_FLAG){

        //   }
        // }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
      }

      // Write Csub to device memory
      // Each thread writes one element
      for (Idx storeLoopIdx = 0; storeLoopIdx < (COARSEN_FACTOR_2_FLAG ? 4 : 1);
           storeLoopIdx++) {
        Idx thIdxRow = thIdxRow_initial + storeLoopIdx * (SHMEM_BLOCK_SIZE / 4);
        Idx thIdxFeat = thIdxFeat_initial;
        if constexpr (COARSEN_FACTOR_2_FLAG) {
          if (storeLoopIdx == 1) {
            Cvalue = Cvalue_1;
          } else if (storeLoopIdx == 2) {
            Cvalue = Cvalue_2;
          } else if (storeLoopIdx == 3) {
            Cvalue = Cvalue_3;
          }
        }
        if constexpr (OuterProductFlag) {
          bool WriteCInRangeFlag =
              blockRow * SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols &&
              blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols;
          // printf("blockRow %d blockFeat %d thIdxRow %d thIdxFeat %d
          // CWriteFlag %d \n", blockRow, blockFeat, thIdxRow, thIdxFeat,
          // WriteCInRangeFlag);
          if constexpr (DoInnerProductSwitch <= 1) {
            if (WriteCInRangeFlag) {
              // NB: this indexing scheme works for both cases whether num_head
              // is
              // 1
              atomicAdd(
                  &C[idx_head * num_A_cols /*A is transposed in the fly*/ *
                         num_B_cols +
                     (blockRow * SHMEM_BLOCK_SIZE + thIdxRow) * num_B_cols +
                     blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat],
                  Cvalue);
            }
          }
        } else {
          bool WriteCInRangeFlag =
              thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
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
            //        thIdxRow + blockRow * BLOCK_SIZE, C_scatter_list[thIdxRow
            //        + blockRow * BLOCK_SIZE], blockFeat * BLOCK_SIZE +
            //        thIdxFeat, idx_head, num_heads, num_B_cols, numARows);
            if constexpr (DoInnerProductSwitch <= 1) {
              atomicAdd(
                  &C[C_scatter_list[thIdxRow + blockRow * SHMEM_BLOCK_SIZE +
                                    blockRowJobEntryBeg] *
                         num_B_cols * num_heads +
                     idx_head * num_B_cols + blockFeat * SHMEM_BLOCK_SIZE +
                     thIdxFeat],
                  Cvalue);
            }
            if constexpr (DoInnerProductSwitch > 0) {
              // TODO: we may hide the global mem read latency by moving input
              // node feat load ahead at the cost of more shmem (copy async) or
              // more registers use
              atomicAdd(&inner_product[inner_product_term_gather_list
                                               [thIdxRow +
                                                blockRow * SHMEM_BLOCK_SIZE +
                                                blockRowJobEntryBeg] *
                                           num_heads +
                                       idx_head],
                        Cvalue *
                            input_node_feat_for_inner_product
                                [C_scatter_list[thIdxRow +
                                                blockRow * SHMEM_BLOCK_SIZE +
                                                blockRowJobEntryBeg] *
                                     num_B_cols * num_heads +
                                 idx_head * num_B_cols +
                                 blockFeat * SHMEM_BLOCK_SIZE + thIdxFeat]);
            }
            // atomicAdd(&GetRowMajorElement<Idx, IdxPtr, ScatterCFlag,
            //                               AdvancedScatterCFlag>(
            //               C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
            //               unique_srcs_and_dests_node_indices, idx_relation,
            //               thIdxRow + blockRow * BLOCK_SIZE +
            //               blockRowJobEntryBeg, C_num_head_one_flag ? 0 :
            //               idx_head, blockFeat * BLOCK_SIZE + thIdxFeat,
            //               C_num_head_one_flag ? 1 : num_heads, num_B_cols),
            //           Cvalue);
          }
        }
      }
    }
  }
};

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_RGCNMatmulNoScatterGatherListFwProp(
    float* node_feat_input, float* weights, float* linear_projected_node_feat,
    float* edge_norm, IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int* accum_num_blocks_per_relation, Idx num_relations, Idx input_dim,
    Idx output_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, false, false, 0, false, false>::
      exec_function(
          node_feat_input, &weights[idx_relation * input_dim * output_dim],
          linear_projected_node_feat, edge_norm, nullptr, nullptr,
          separate_coo_row_idx, separate_coo_col_idx, separate_coo_eids,
          idx_relation,
          separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          separate_coo_rel_ptrs[idx_relation], input_dim, output_dim, 1);
}

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_RGCNMatmulNoScatterGatherListDeltaWeightBckProp(
    float* node_feat_input, float* delta_linear_projected_node_feat,
    float* delta_weights, float* edge_norm, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx delta_output_dim, Idx delta_input_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, false, false, 0, false, false>::
      exec_function(
          node_feat_input, delta_linear_projected_node_feat,
          &delta_weights[idx_relation * delta_output_dim * delta_input_dim],
          edge_norm, nullptr, nullptr, separate_coo_row_idx,
          separate_coo_col_idx, separate_coo_eids, idx_relation,
          separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          separate_coo_rel_ptrs[idx_relation], delta_output_dim,
          delta_input_dim, 1);
}

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_RGCNMatmulNoScatterGatherListDeltaNodeFeatBckProp(
    float* delta_linear_projected_node_feat, float* weights_transposed,
    float* delta_node_feat_input, float* edge_norm, float* grad_edge_norm,
    float* input_node_feat_for_grad_norm, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx delta_output_dim, Idx delta_input_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, false, false, 1, true, false>::
      exec_function(delta_linear_projected_node_feat,
                    &weights_transposed[idx_relation * delta_output_dim *
                                        delta_input_dim],
                    delta_node_feat_input, edge_norm, grad_edge_norm,
                    input_node_feat_for_grad_norm, separate_coo_row_idx,
                    separate_coo_col_idx, separate_coo_eids, idx_relation,
                    separate_coo_rel_ptrs[idx_relation + 1] -
                        separate_coo_rel_ptrs[idx_relation],
                    accum_num_blocks_per_relation[idx_relation],
                    (accum_num_blocks_per_relation[idx_relation + 1] -
                     accum_num_blocks_per_relation[idx_relation]),
                    separate_coo_rel_ptrs[idx_relation], delta_output_dim,
                    delta_input_dim, 1);
}

// delta weight: A feat_input, B delta_feat_out, C delta_weight,
// delta in: A delta_feat_out, B weight_transposed, C delta_feat_in,

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_HGTMessageGenerationAndAccumulationFwProp(
    float* node_feat_input, float* weights, float* linear_projected_node_feat,
    float* edge_norm, /*float* relation_pri, */ IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx input_dim, Idx output_dim, Idx num_heads) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, false, 0, false, false>::
      exec_function(node_feat_input,
                    &weights[idx_relation * num_heads * input_dim * output_dim],
                    linear_projected_node_feat, edge_norm, nullptr, nullptr,
                    separate_coo_row_idx, separate_coo_col_idx,
                    separate_coo_eids, idx_relation,
                    separate_coo_rel_ptrs[idx_relation + 1] -
                        separate_coo_rel_ptrs[idx_relation],
                    accum_num_blocks_per_relation[idx_relation],
                    (accum_num_blocks_per_relation[idx_relation + 1] -
                     accum_num_blocks_per_relation[idx_relation]),
                    separate_coo_rel_ptrs[idx_relation], input_dim, output_dim,
                    num_heads);
}

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_HGTMessageGenerationAndAccumulationDeltaWeightBckProp(
    float* node_feat_input, float* delta_linear_projected_node_feat,
    float* delta_weights, float* edge_norm, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx input_dim, Idx delta_output_dim, Idx num_heads) {
  // TODO: block assignment scheme might be different when OuterProductFlag ==
  // True
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, true, 0, false, false>::
      exec_function(node_feat_input, delta_linear_projected_node_feat,
                    &delta_weights[idx_relation * num_heads * input_dim *
                                   delta_output_dim],
                    edge_norm, nullptr, nullptr, separate_coo_row_idx,
                    separate_coo_col_idx, separate_coo_eids, idx_relation,
                    separate_coo_rel_ptrs[idx_relation + 1] -
                        separate_coo_rel_ptrs[idx_relation],
                    accum_num_blocks_per_relation[idx_relation],
                    (accum_num_blocks_per_relation[idx_relation + 1] -
                     accum_num_blocks_per_relation[idx_relation]),
                    separate_coo_rel_ptrs[idx_relation], input_dim,
                    delta_output_dim, num_heads);
}

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void
HET_HGTMessageGenerationAndAccumulationDeltaNodeFeatInputBckProp(
    float* delta_linear_projected_node_feat, float* weights_transposed,
    float* delta_node_feat_input, float* edge_norm, float* grad_edge_norm,
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int* accum_num_blocks_per_relation, Idx num_relations, Idx delta_output_dim,
    Idx delta_input_dim, Idx num_heads) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, false, 1, false, false>::
      exec_function(delta_linear_projected_node_feat,
                    &weights_transposed[idx_relation * num_heads *
                                        delta_output_dim * delta_input_dim],
                    delta_node_feat_input, edge_norm, grad_edge_norm, nullptr,
                    separate_coo_row_idx, separate_coo_col_idx,
                    separate_coo_eids, idx_relation,
                    separate_coo_rel_ptrs[idx_relation + 1] -
                        separate_coo_rel_ptrs[idx_relation],
                    accum_num_blocks_per_relation[idx_relation],
                    (accum_num_blocks_per_relation[idx_relation + 1] -
                     accum_num_blocks_per_relation[idx_relation]),
                    separate_coo_rel_ptrs[idx_relation], delta_output_dim,
                    delta_input_dim, num_heads);
}

template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_HGTFusedAttnScoreFwProp(
    float* applied_klinear_node_features, float* applied_qlinear_node_features,
    float* attn_score_weight, float* attn_score_inner_product,
    float* unnormalized_attn_score, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int* accum_num_blocks_per_relation,
    Idx num_relations, Idx fw_input_dim_per_head, Idx fw_output_dim_per_head,
    Idx num_heads) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  // NB: should be mode 1 since we need to output inner product for bck prop use
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, false, 1, false, true>::
      exec_function(
          applied_klinear_node_features,
          &attn_score_weight[idx_relation * num_heads * fw_output_dim_per_head *
                             fw_input_dim_per_head],
          attn_score_inner_product, nullptr, unnormalized_attn_score,
          applied_qlinear_node_features, separate_coo_row_idx,
          separate_coo_col_idx, separate_coo_eids, idx_relation,
          separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          separate_coo_rel_ptrs[idx_relation], fw_input_dim_per_head,
          fw_output_dim_per_head, num_heads);
}

// delta_k = delta_inner_product*weight_transposed =
// delta_attn_score*q*weight_transposed
template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_HGTFusedAttnScoreDeltaKVectBckProp(
    float* applied_qlinear_node_features, float* attn_score_weight_transposed,
    float* delta_applied_klinear_node_features, float* grad_attn_score,
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int* accum_num_blocks_per_relation, Idx num_relations,
    Idx fw_input_dim_per_head, Idx fw_output_dim_per_head, Idx num_heads) {
  // edge_norm is delta_attn_score

  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, false, 0, false, false>::
      exec_function(applied_qlinear_node_features,
                    &attn_score_weight_transposed[idx_relation * num_heads *
                                                  fw_output_dim_per_head *
                                                  fw_input_dim_per_head],
                    delta_applied_klinear_node_features, grad_attn_score,
                    nullptr, nullptr, separate_coo_row_idx,
                    separate_coo_col_idx, separate_coo_eids, idx_relation,
                    separate_coo_rel_ptrs[idx_relation + 1] -
                        separate_coo_rel_ptrs[idx_relation],
                    accum_num_blocks_per_relation[idx_relation],
                    (accum_num_blocks_per_relation[idx_relation + 1] -
                     accum_num_blocks_per_relation[idx_relation]),
                    separate_coo_rel_ptrs[idx_relation], fw_output_dim_per_head,
                    fw_input_dim_per_head, num_heads);
}

// delta_weight=delta_attn_score*inner_product_transposed
template <bool COARSEN_FACTOR_2_FLAG, int BLOCK_SIZE, typename Idx,
          typename IdxPtr>
__global__ void HET_HGTFusedAttnScoreDeltaWeightBckProp(
    float* applied_klinear_node_features, float* applied_qlinear_node_features,
    float* attn_score_weight, float* grad_attn_score,
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int* accum_num_blocks_per_relation, Idx num_relations,
    Idx fw_input_dim_per_head, Idx fw_output_dim_per_head, Idx num_heads) {
  // edge_norm is delta_attn_score

  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int*>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<false, COARSEN_FACTOR_2_FLAG, BLOCK_SIZE, Idx,
                                 IdxPtr, true, true, 0, false, false>::
      exec_function(
          applied_klinear_node_features, applied_qlinear_node_features,
          &attn_score_weight[idx_relation * num_heads * fw_output_dim_per_head *
                             fw_input_dim_per_head],
          grad_attn_score, nullptr, nullptr, separate_coo_row_idx,
          separate_coo_col_idx, separate_coo_eids, idx_relation,
          separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          separate_coo_rel_ptrs[idx_relation], fw_input_dim_per_head,
          fw_output_dim_per_head, num_heads);
}
