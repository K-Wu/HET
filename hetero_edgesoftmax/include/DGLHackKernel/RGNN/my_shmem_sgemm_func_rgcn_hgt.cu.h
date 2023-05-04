#pragma once
#include <cuda_runtime.h>
// #include "cuda.h"
#include "my_shmem_sgemm_func_rgcn_hgt_functor.cu.h"
#include "utils.cu.h"

// NB: KWU: generalize COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y to
// THREAD_BLOCK_SIZE_X, THREAD_BLOCK_SIZE_Y
// NB: KWU: split SHMEM_BLOCK_SIZE to TILE_SIZE_X, TILE_SIZE_Y, TILE_SIZE_K
// NB: KWU: add register tiling (left matrix uses shmem tiling and right
// matrix uses register tiling) X, Y stands for the grid/block dimension name
// and corresponds to B column and A row, respectively
template <int THREAD_BLOCK_DIM_X, int THREAD_BLOCK_DIM_Y,
          int SHMEM_BLOCK_SIZE_X, int SHMEM_BLOCK_SIZE_Y,
          int SHMEM_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
          bool HGT_INSTEAD_OF_RGCN_FLAG, bool OuterProductFlag,
          int DoInnerProductSwitch,
          bool InnerProductGatherListNodeInsteadOfEdge, bool NoEdgeNormFlag>
class _simplified_basic_MatMulKernel<
    false, THREAD_BLOCK_DIM_X, THREAD_BLOCK_DIM_Y, SHMEM_BLOCK_SIZE_X,
    SHMEM_BLOCK_SIZE_Y, SHMEM_BLOCK_SIZE_K, Idx, IdxPtr,
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
  __device__ __forceinline__ static void execute_function(
      float *A, float *B, float *C, float *edge_norm, float *inner_product,
      float *input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
      IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids, Idx idx_relation,
      Idx numARows, Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
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

    // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
    // threadIdx if in a mega-kernel

    constexpr bool RIGHT_REG_TILED_FLAG = THREAD_BLOCK_DIM_Y == 1;

    constexpr int COARSEN_DIVISOR_FACTOR =
        (SHMEM_BLOCK_SIZE_Y / THREAD_BLOCK_DIM_Y) *
        (SHMEM_BLOCK_SIZE_X / THREAD_BLOCK_DIM_X);
    static_assert(SHMEM_BLOCK_SIZE_Y % THREAD_BLOCK_DIM_Y == 0, "");
    static_assert(SHMEM_BLOCK_SIZE_X % THREAD_BLOCK_DIM_X == 0, "");
    constexpr int COARSEN_DIVISOR_FACTOR_LOAD_A =
        SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_X / THREAD_BLOCK_DIM_X /
        THREAD_BLOCK_DIM_Y;
    static_assert((SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_X) %
                          (THREAD_BLOCK_DIM_X * THREAD_BLOCK_DIM_Y) ==
                      0,
                  "");
    constexpr int COARSEN_DIVISOR_FACTOR_LOAD_B =
        SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_Y / THREAD_BLOCK_DIM_X /
        THREAD_BLOCK_DIM_Y;
    static_assert((SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_Y) %
                          (THREAD_BLOCK_DIM_X * THREAD_BLOCK_DIM_Y) ==
                      0,
                  "");
    if constexpr (!HGT_INSTEAD_OF_RGCN_FLAG) {
      // assuming this case is RGCN and there is no multiple head
      assert((blockDim.z == 1));
    }  // otherwise assuming HGT
    // TODO: use int for blockIdx threadIdx related variables
    // NB: this scheme does not support num_heads > int_max
    int idx_head = blockIdx.z % num_heads;
    IdxPtr A_gather_list;
    IdxPtr C_scatter_list;
    IdxPtr B_gather_list;
    IdxPtr inner_product_term_scatter_list;
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
        inner_product_term_scatter_list = separate_coo_col_idx;
      } else {
        inner_product_term_scatter_list = separate_coo_eids;
      }
    }
    // Idx blockRow = blockIdx.y - blockIdxAlongRowBeg;
    int blockFeat = blockIdx.x;  // when OuterProductFlag==True, it is in [0,
                                 // output_dim//num_heads)

    int blockRowLoopBeg, blockRowLoopEnd, blockRowLoopInc;
    if constexpr (OuterProductFlag) {
      blockRowLoopBeg =
          blockIdx.y;  // [0, input_dim) // check my_shmem_sgemm_func.cu.h NB on
                       // why -blockIdxAlongRowBeg bias is not applied here but
                       // applied to the m loop
      blockRowLoopEnd = blockIdx.y + 1;
      blockRowLoopInc = 1;
    } else {
      blockRowLoopBeg = blockIdx.y - blockIdxAlongRowBeg;
      blockRowLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE_Y);
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
      // TODO: KWU: use shared memory to store innerproductterm
      float InnerProductTerm[COARSEN_DIVISOR_FACTOR] =
          {};  // zero initialization

      // Thread row and column within Csub
      int thIdxRow_initial = threadIdx.y;
      int thIdxFeat_initial = threadIdx.x;
      if constexpr (COARSEN_DIVISOR_FACTOR > 1) {
        int thIdx = threadIdx.y * THREAD_BLOCK_DIM_X + threadIdx.x;
        thIdxRow_initial = thIdx / SHMEM_BLOCK_SIZE_X;
        thIdxFeat_initial = thIdx % SHMEM_BLOCK_SIZE_X;
      }
      // Loop over all the sub-matrices of A and B that are
      // required to compute Csub
      // Multiply each pair of sub-matrices together
      // and accumulate the results

      // load inner product term in advance
      if constexpr (DoInnerProductSwitch > 0) {
        for (int idx_coarsen_factor = 0;
             idx_coarsen_factor < COARSEN_DIVISOR_FACTOR;
             idx_coarsen_factor++) {
          static_assert(SHMEM_BLOCK_SIZE_Y >= COARSEN_DIVISOR_FACTOR, "");
          int thIdxRow = thIdxRow_initial +
                         idx_coarsen_factor *
                             (SHMEM_BLOCK_SIZE_Y / COARSEN_DIVISOR_FACTOR);
          int thIdxFeat = thIdxFeat_initial;
          bool WriteCInRangeFlag =
              thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y < numARows &&
              blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols;
          InnerProductTerm[idx_coarsen_factor] =
              WriteCInRangeFlag
                  ? input_node_feat_for_inner_product
                        [C_scatter_list[thIdxRow +
                                        blockRow * SHMEM_BLOCK_SIZE_Y +
                                        blockRowJobEntryBeg] *
                             num_B_cols * num_heads +
                         idx_head * num_B_cols +
                         blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat]
                  : 0.0f;
        }
      }

      int mLoopBeg, mLoopEnd, mLoopInc;
      if constexpr (OuterProductFlag) {
        int blockAssignmentIdx = blockIdx.z / num_heads;
        mLoopBeg = blockAssignmentIdx - blockIdxAlongRowBeg;
        mLoopEnd = ceil_div<>(numARows, (Idx)SHMEM_BLOCK_SIZE_Y);
        mLoopInc = strideNumBlocksAlongRow;
      } else {
        mLoopBeg = 0;
        mLoopEnd = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE_K);
        mLoopInc = 1;
      }
      for (int m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[SHMEM_BLOCK_SIZE_Y][SHMEM_BLOCK_SIZE_K];
        __shared__ float Bs[RIGHT_REG_TILED_FLAG ? 1 : SHMEM_BLOCK_SIZE_K]
                           [RIGHT_REG_TILED_FLAG ? 1 : SHMEM_BLOCK_SIZE_X];
        float Bs_reg[RIGHT_REG_TILED_FLAG
                         ? SHMEM_BLOCK_SIZE_K
                         : 1];  // Bs_reg[e] == Bs[e][thIdxFeat_initial];

        // Get sub-matrix Bsub of B
        // Load Asub and Bsub from device memory to
        // shared memory Each thread loads one element of each sub-matrix

        // Get sub-matrix Asub of A
        // for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR;
        //      loadLoopIdx++) {
        // int thIdxRow =
        //     thIdxRow_initial +
        //     loadLoopIdx * (SHMEM_BLOCK_SIZE_Y / COARSEN_DIVISOR_FACTOR);
        // int thIdxFeat = thIdxFeat_initial;
        if constexpr (OuterProductFlag) {
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_A;
               loadLoopIdx++) {
            // NB: in outer product, m and y are interchanged, and the loading
            // scheme is a transpose in the fly. that is why both thIdxRow and m
            // need to be used during indexing the row
            int SHMEM_BLOCK_DIM_Y_PER_LOAD_A_OUTER_PRODUCT =
                THREAD_BLOCK_DIM_X * THREAD_BLOCK_DIM_Y / SHMEM_BLOCK_SIZE_K;
            int thIdxRow_A_outer_product =
                (threadIdx.y * THREAD_BLOCK_DIM_X + threadIdx.x) %
                    SHMEM_BLOCK_DIM_Y_PER_LOAD_A_OUTER_PRODUCT +
                loadLoopIdx * (SHMEM_BLOCK_SIZE_Y / COARSEN_DIVISOR_FACTOR);
            static_assert(SHMEM_BLOCK_SIZE_Y % COARSEN_DIVISOR_FACTOR == 0, "");
            int thIdxFeat_A_outer_product =
                (threadIdx.y * THREAD_BLOCK_DIM_X + threadIdx.x) /
                SHMEM_BLOCK_DIM_Y_PER_LOAD_A_OUTER_PRODUCT;
            As[thIdxRow_A_outer_product][thIdxFeat_A_outer_product] =
                (thIdxFeat_A_outer_product +
                         (m)*SHMEM_BLOCK_SIZE_K <  // + blockRowJobEntryBeg <
                     numARows &&
                 blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow_A_outer_product <
                     num_A_cols)
                    ? A[A_gather_list[thIdxFeat_A_outer_product +
                                      (m)*SHMEM_BLOCK_SIZE_K +
                                      blockRowJobEntryBeg] *
                            num_A_cols * num_heads +
                        num_A_cols * idx_head +
                        (blockRow * SHMEM_BLOCK_SIZE_Y +
                         thIdxRow_A_outer_product)] *
                          (NoEdgeNormFlag
                               ? 1.0f
                               : edge_norm[separate_coo_eids
                                                   [thIdxFeat_A_outer_product +
                                                    (m)*SHMEM_BLOCK_SIZE_K +
                                                    blockRowJobEntryBeg] *
                                               num_heads +
                                           idx_head])
                    : 0.0f;
          }
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_B;
               loadLoopIdx++) {
            // NB: KWU: for outerproduct, the m loop variable is on the K
            // dimension
            int thIdxRow = thIdxRow_initial +
                           loadLoopIdx * (SHMEM_BLOCK_SIZE_K /
                                          COARSEN_DIVISOR_FACTOR_LOAD_B);
            int thIdxFeat = thIdxFeat_initial;
            float value_to_load =
                ((m)*SHMEM_BLOCK_SIZE_K + thIdxRow <  //+ blockRowJobEntryBeg <
                     numARows &&
                 blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols)
                    ? B[B_gather_list[(m)*SHMEM_BLOCK_SIZE_K + thIdxRow +
                                      blockRowJobEntryBeg] *
                            num_B_cols * num_heads +
                        num_B_cols * idx_head +
                        (blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat)]
                    : 0.0f;
            if constexpr (RIGHT_REG_TILED_FLAG) {
              Bs_reg[thIdxRow] = value_to_load;
            } else {
              Bs[thIdxRow][thIdxFeat] = value_to_load;
            }
          }
        } else {
          // printf("blockRow %d blockFeat %d thIdxRow %d thIdxFeat %d
          // AloadFlag %d BloadFlag %d \n", blockRow, blockFeat, thIdxRow,
          // thIdxFeat, (thIdxRow + blockRow * SHMEM_BLOCK_SIZE < numARows &&
          //      m * SHMEM_BLOCK_SIZE + thIdxFeat < num_A_cols), (m *
          //      SHMEM_BLOCK_SIZE + thIdxRow < num_A_cols && blockFeat *
          //      SHMEM_BLOCK_SIZE + thIdxFeat < num_B_cols));
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_A;
               loadLoopIdx++) {
            int thIdxRow = thIdxRow_initial +
                           loadLoopIdx * (SHMEM_BLOCK_SIZE_Y /
                                          COARSEN_DIVISOR_FACTOR_LOAD_A);
            int thIdxFeat = thIdxFeat_initial;
            As[thIdxRow][thIdxFeat] =
                (thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y < numARows &&
                 m * SHMEM_BLOCK_SIZE_K + thIdxFeat < num_A_cols)
                    ? A[A_gather_list[thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y +
                                      blockRowJobEntryBeg] *
                            num_A_cols * num_heads +
                        num_A_cols * idx_head +
                        (thIdxFeat + m * SHMEM_BLOCK_SIZE_K)] *
                          (NoEdgeNormFlag
                               ? 1.0f
                               : edge_norm
                                     [separate_coo_eids[thIdxRow +
                                                        blockRow *
                                                            SHMEM_BLOCK_SIZE_Y +
                                                        blockRowJobEntryBeg] *
                                          num_heads +
                                      idx_head])
                    : 0.0f;
          }
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

          // In this case, B is weight and therefore the indexing scheme is
          // different, i.e., [num_heads, num_B_rows_feat, num_B_cols_feat]
          // instead of [num_nodes|edges, num_heads, num_feats]
          // FIXME: the n_head term in the following indexing scheme may not
          // be correct.
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_B;
               loadLoopIdx++) {
            int thIdxRow = thIdxRow_initial +
                           loadLoopIdx * (SHMEM_BLOCK_SIZE_K /
                                          COARSEN_DIVISOR_FACTOR_LOAD_B);
            int thIdxFeat = thIdxFeat_initial;

            float value_to_load =
                (m * SHMEM_BLOCK_SIZE_K + thIdxRow < num_A_cols &&
                 blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols)
                    ? B[(m * SHMEM_BLOCK_SIZE_K + thIdxRow) * num_B_cols +
                        idx_head * num_B_cols * num_A_cols +
                        (blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat)]
                    : 0.0f;
            if constexpr (RIGHT_REG_TILED_FLAG) {
              Bs_reg[thIdxFeat] = value_to_load;
            } else {
              Bs[thIdxRow][thIdxFeat] = value_to_load;
            }
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

        // TODO: KWU: switch to cg::sync
        __syncthreads();
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        // Multiply Asub and Bsub together
        for (int e = 0; e < SHMEM_BLOCK_SIZE_K; ++e) {
          for (int idx_coarsen_factor = 0;
               idx_coarsen_factor < COARSEN_DIVISOR_FACTOR;
               ++idx_coarsen_factor) {
            float right_operand;
            if constexpr (RIGHT_REG_TILED_FLAG) {
              right_operand = Bs_reg[e];
            } else {
              right_operand = Bs[e][thIdxFeat_initial];
            }
            Cvalue[idx_coarsen_factor] +=
                As[thIdxRow_initial +
                   idx_coarsen_factor *
                       (SHMEM_BLOCK_SIZE_Y / COARSEN_DIVISOR_FACTOR)][e] *
                right_operand;
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
      for (int storeLoopIdx = 0; storeLoopIdx < COARSEN_DIVISOR_FACTOR;
           storeLoopIdx++) {
        int thIdxRow =
            thIdxRow_initial +
            storeLoopIdx * (SHMEM_BLOCK_SIZE_Y / COARSEN_DIVISOR_FACTOR);
        int thIdxFeat = thIdxFeat_initial;
        if constexpr (OuterProductFlag) {
          bool WriteCInRangeFlag =
              blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow < num_A_cols &&
              blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols;

          // printf("idx_relation %d blockRow %d blockFeat %d thIdxRow %d
          // thIdxFeat %d CWriteFlag %d \n", idx_relation, blockRow, blockFeat,
          // thIdxRow, thIdxFeat, WriteCInRangeFlag);
          if constexpr (DoInnerProductSwitch > 0) {
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                (DoInnerProductSwitch > 0) && OuterProductFlag,
                "DoInnerProductSwitch > 0 && OuterProductFlag");
          }
          if (WriteCInRangeFlag) {
            // NB: this indexing scheme works for both cases whether num_head
            // is
            // 1

            atomicAdd(
                &C[idx_head * num_A_cols /*A is transposed in the fly*/ *
                       num_B_cols +
                   (blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow) * num_B_cols +
                   blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat],
                Cvalue[storeLoopIdx]);
          }
        } else {
          bool WriteCInRangeFlag =
              thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y < numARows &&
              blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols;
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
          // if constexpr (DoInnerProductSwitch>0){
          //     if(threadIdx.x%16==0){
          //     if (WriteCInRangeFlag){
          //       printf("row_id %d, col_id %d, Cvalue %f, InnerProductTerm %f,
          //       Write True, SHMEM %d, thIdxRow %d, thIdxFeat %d, threadIdx.x
          //       %d, threadIdx.y %d\n", (int) thIdxRow +
          //                                       blockRow * SHMEM_BLOCK_SIZE +
          //                                       (int) blockRowJobEntryBeg,
          //                                       (int) blockFeat *
          //                                       SHMEM_BLOCK_SIZE +
          //            thIdxFeat,(float)Cvalue, (float)InnerProductTerm,
          //            SHMEM_BLOCK_SIZE, thIdxRow, thIdxFeat, threadIdx.x,
          //            threadIdx.y);
          //     }
          //     else{
          //       printf("row_id %d, col_id %d, Cvalue %f, InnerProductTerm %f,
          //       Write False, SHMEM %d, thIdxRow %d, thIdxFeat %d, threadIdx.x
          //       %d, threadIdx.y %d\n",  (int) thIdxRow +
          //                                       blockRow * SHMEM_BLOCK_SIZE +
          //       (int) blockRowJobEntryBeg,  (int) blockFeat *
          //       SHMEM_BLOCK_SIZE +
          //            thIdxFeat,(float)Cvalue, (float)InnerProductTerm,
          //            SHMEM_BLOCK_SIZE, thIdxRow, thIdxFeat, threadIdx.x,
          //            threadIdx.y);
          //     }
          //     }
          //   __syncwarp();
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
                  &C[C_scatter_list[thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y +
                                    blockRowJobEntryBeg] *
                         num_B_cols * num_heads +
                     idx_head * num_B_cols + blockFeat * SHMEM_BLOCK_SIZE_X +
                     thIdxFeat],
                  Cvalue[storeLoopIdx]);
            }
            // NB: warp-level reduction here
            if constexpr (DoInnerProductSwitch > 0) {
              // TODO: we may hide the global mem read latency by moving input
              // node feat load ahead at the cost of more shmem (copy async) or
              // more registers use

              // atomicAdd(&inner_product[inner_product_term_scatter_list
              //                                  [thIdxRow +
              //                                   blockRow * SHMEM_BLOCK_SIZE +
              //                                   blockRowJobEntryBeg] *
              //                              num_heads +
              //                          idx_head],
              //           Cvalue * InnerProductTerm);

              unsigned int mask_size =
                  32 > SHMEM_BLOCK_SIZE_X ? SHMEM_BLOCK_SIZE_X : 32;
              unsigned int mask = ((1ULL) << mask_size) - 1;
              float product_sum =
                  Cvalue[storeLoopIdx] * InnerProductTerm[storeLoopIdx];
              for (int offset = mask_size / 2; offset > 0; offset /= 2) {
                product_sum += __shfl_xor_sync(mask, product_sum, offset);
              }
              if (threadIdx.x % mask_size == 0) {
                atomicAdd(
                    &inner_product[inner_product_term_scatter_list
                                           [thIdxRow +
                                            blockRow * SHMEM_BLOCK_SIZE_Y +
                                            blockRowJobEntryBeg] *
                                       num_heads +
                                   idx_head],
                    product_sum);
              }
            }
          }
        }
      }
    }
  }
};

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void __launch_bounds__(256, 3)
    HET_RGCNMatmulNoScatterGatherListFwProp(
        float *node_feat_input, float *weights,
        float *linear_projected_node_feat, float *edge_norm,
        IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
        IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
        int *accum_num_blocks_per_relation, Idx num_relations, Idx input_dim,
        Idx output_dim) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, false,
      false, 0, false, false>::
      execute_function(
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

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void __launch_bounds__(256, 3)
    HET_RGCNMatmulNoScatterGatherListDeltaWeightBckProp(
        float *node_feat_input, float *delta_linear_projected_node_feat,
        float *delta_weights, float *edge_norm, IdxPtr separate_coo_row_idx,
        IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
        IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
        Idx num_relations, Idx delta_output_dim, Idx delta_input_dim) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.z;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, false,
      true, 0, false, false>::
      execute_function(
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

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void __launch_bounds__(256, 3)
    HET_RGCNMatmulNoScatterGatherListDeltaNodeFeatBckProp(
        float *delta_linear_projected_node_feat, float *weights_transposed,
        float *delta_node_feat_input, float *edge_norm, float *grad_edge_norm,
        float *input_node_feat_for_grad_norm, IdxPtr separate_coo_row_idx,
        IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
        IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
        Idx num_relations, Idx delta_output_dim, Idx delta_input_dim) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, false,
      false, 0, true, false>::
      execute_function(delta_linear_projected_node_feat,
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

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void __launch_bounds__(256, 3)
    HET_HGTMessageGenerationAndAccumulationFwProp(
        float *node_feat_input, float *weights,
        float *linear_projected_node_feat, float *edge_norm,
        /*float* relation_pri, */ IdxPtr separate_coo_row_idx,
        IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
        IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
        Idx num_relations, Idx input_dim, Idx output_dim, int num_heads) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      false, 0, false, false>::
      execute_function(
          node_feat_input,
          &weights[idx_relation * num_heads * input_dim * output_dim],
          linear_projected_node_feat, edge_norm, nullptr, nullptr,
          separate_coo_row_idx, separate_coo_col_idx, separate_coo_eids,
          idx_relation,
          separate_coo_rel_ptrs[idx_relation + 1] -
              separate_coo_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          separate_coo_rel_ptrs[idx_relation], input_dim, output_dim,
          num_heads);
}

// FIXME: no coarsening for delta weight
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HET_HGTMessageGenerationAndAccumulationDeltaWeightBckProp(
    float *node_feat_input, float *delta_linear_projected_node_feat,
    float *delta_weights, float *edge_norm, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
    Idx num_relations, Idx input_dim, Idx delta_output_dim, int num_heads) {
  // TODO: block assignment scheme might be different when OuterProductFlag ==
  // True
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      true, 0, false, false>::
      execute_function(node_feat_input, delta_linear_projected_node_feat,
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

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void
HET_HGTMessageGenerationAndAccumulationDeltaNodeFeatInputBckProp(
    float *delta_linear_projected_node_feat, float *weights_transposed,
    float *delta_node_feat_input, float *node_feat_input, float *edge_norm,
    float *grad_edge_norm, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
    Idx num_relations, Idx delta_output_dim, Idx delta_input_dim,
    int num_heads) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      false, 1, false, false>::
      execute_function(delta_linear_projected_node_feat,
                       &weights_transposed[idx_relation * num_heads *
                                           delta_output_dim * delta_input_dim],
                       delta_node_feat_input, edge_norm, grad_edge_norm,
                       node_feat_input, separate_coo_row_idx,
                       separate_coo_col_idx, separate_coo_eids, idx_relation,
                       separate_coo_rel_ptrs[idx_relation + 1] -
                           separate_coo_rel_ptrs[idx_relation],
                       accum_num_blocks_per_relation[idx_relation],
                       (accum_num_blocks_per_relation[idx_relation + 1] -
                        accum_num_blocks_per_relation[idx_relation]),
                       separate_coo_rel_ptrs[idx_relation], delta_output_dim,
                       delta_input_dim, num_heads);
}

template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void __launch_bounds__(256, 3) HET_HGTFusedAttnScoreFwProp(
    float *applied_klinear_node_features, float *applied_qlinear_node_features,
    float *attn_score_weight, float *attn_score_inner_product,
    float *unnormalized_attn_score, IdxPtr separate_coo_row_idx,
    IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids,
    IdxPtr separate_coo_rel_ptrs, int *accum_num_blocks_per_relation,
    Idx num_relations, Idx fw_input_dim_per_head, Idx fw_output_dim_per_head,
    int num_heads) {
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  // NB: should be mode 1 since we need to output inner product for bck prop use
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      false, 1, false, true>::
      execute_function(
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
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HET_HGTFusedAttnScoreDeltaKVectBckProp(
    float *applied_qlinear_node_features, float *attn_score_weight_transposed,
    float *delta_applied_klinear_node_features, float *grad_attn_score,
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int *accum_num_blocks_per_relation, Idx num_relations,
    Idx fw_input_dim_per_head, Idx fw_output_dim_per_head, int num_heads) {
  // edge_norm is delta_attn_score
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      false, 0, false, false>::
      execute_function(applied_qlinear_node_features,
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
                       separate_coo_rel_ptrs[idx_relation],
                       fw_output_dim_per_head, fw_input_dim_per_head,
                       num_heads);
}

// delta_weight=delta_attn_score*inner_product_transposed
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void HET_HGTFusedAttnScoreDeltaWeightBckProp(
    float *applied_klinear_node_features, float *applied_qlinear_node_features,
    float *grad_attn_score_weight, float *grad_attn_score,
    IdxPtr separate_coo_row_idx, IdxPtr separate_coo_col_idx,
    IdxPtr separate_coo_eids, IdxPtr separate_coo_rel_ptrs,
    int *accum_num_blocks_per_relation, Idx num_relations,
    Idx fw_input_dim_per_head, Idx fw_output_dim_per_head, int num_heads) {
  // edge_norm is delta_attn_score
  // TODO: use int instead for idx_block_assignment and idx_relation
  // TODO: KUW: supercede blockIdx/threadIdx with pretended blockIdx and
  // threadIdx if in a mega-kernel
  int idx_block_assignment = blockIdx.z / num_heads;
  int idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _simplified_basic_MatMulKernel<
      false, COARSEN_FACTOR_2_FLAG_X ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      COARSEN_FACTOR_2_FLAG_Y ? SHMEM_BLOCK_SIZE / 2 : SHMEM_BLOCK_SIZE,
      SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, SHMEM_BLOCK_SIZE, Idx, IdxPtr, true,
      true, 0, false, false>::
      execute_function(
          applied_klinear_node_features, applied_qlinear_node_features,
          &grad_attn_score_weight[idx_relation * num_heads *
                                  fw_output_dim_per_head *
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
