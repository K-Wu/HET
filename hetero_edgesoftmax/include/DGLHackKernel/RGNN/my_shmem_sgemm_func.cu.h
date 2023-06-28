#pragma once
#include <cuda_runtime.h>
// #include "cuda.h"
#include "kernel_enums.h"
#include "macros.h"
#include "my_shmem_sgemm_func_functor.cu.h"
#include "utils.cu.h"

// NB: On atomicFlag Notice that right now both deltaweight (i.e.
// OuterProductFlag) and (deltanode calculation where there might be parallel
// updater) requires atomicflag, and seems reasonable. Is there any optimization
// possible?
// TODO: use gather instead of scatter to remove atomicFlag
template <bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
          int THREADING_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_X,
          int SHMEM_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_K, bool OuterProductFlag,
          MySGEMMGatherKind AGatherKind, MySGEMMGatherKind BGatherKind,
          MySGEMMGatherKind CScatterKind, bool AtomicUpdateFlag, typename Idx,
          typename IdxPtr, MySGEMMNumHeadKind numHeadKind,
          CompactAsOfNodeKind compactKind>
class _basic_MatMulKernel<
    RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
    SHMEM_BLOCK_SIZE_X, SHMEM_BLOCK_SIZE_Y, SHMEM_BLOCK_SIZE_K,
    OuterProductFlag, AGatherKind, BGatherKind, CScatterKind, AtomicUpdateFlag,
    Idx, IdxPtr, numHeadKind, compactKind> {
 public:
  // vanilla tiled shmem gemm code from
  // http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
  //@@ Example of grid and block configuration
  //	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 dimGrid(B.width / dimBlock.x, A.height /
  // dimBlock.y, num_heads * num_partitions_along_Acol_Brow );
  // MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
  // C = A * B (all are row-major matrices)
  __device__ __forceinline__ static void execute_function(
      float *A, float *B, float *C, IdxPtr A_gather_list, IdxPtr B_gather_list,
      IdxPtr C_scatter_list,
      // TODO: remove etype_mapper_data as the two_order acccess
      // scheme is never used.
      const ETypeMapperData<Idx, compactKind> etype_mapper_data,
      Idx idx_relation, Idx numARows, Idx blockIdxAlongRowBeg,
      Idx strideNumBlocksAlongRow, Idx blockRowJobEntryBeg, Idx num_A_cols,
      Idx num_B_cols, int num_heads) {
    // num_B_cols is output_dim//num_heads as forward propagation weight,
    // output_dim//num_heads as backward propagation weight, and in_feat_dim as
    // features or delta features. num_A_cols is input_dim as forward
    // propagation input feature, output_dim//num_heads as delta input feature
    constexpr bool BWeightInsteadOfFeatureFlag = !OuterProductFlag;

    constexpr bool A_num_head_one_flag =
        (numHeadKind == MySGEMMNumHeadKind::AssertAllAreOnes ||
         numHeadKind == MySGEMMNumHeadKind::AssertANumIsOne);
    constexpr bool B_num_head_one_flag =
        (numHeadKind == MySGEMMNumHeadKind::AssertAllAreOnes);
    constexpr bool C_num_head_one_flag =
        (numHeadKind == MySGEMMNumHeadKind::AssertAllAreOnes ||
         numHeadKind == MySGEMMNumHeadKind::AssertCNumIsOne);

    // NB: when OuterProductFlag is true and the model is RGAT, num_heads of the
    // input features is always 1 and the other is num_heads. In case of RGCN,
    // both are 1. In case of HGT, both are num_heads except for the kqva linear
    // (both are 1). In the implementation, when OuterProductFlag is true, A is
    // always input and B the gradient output feature. It is safe to pass
    // num_A_cols as in_dim.

    constexpr bool COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT = true;
    //    !RIGHT_REG_TILED_FLAG;
    // if the flag is true, each thread is in charge of a fraction of the output
    // element. Otherwise, each thread is in charge of a fraction of the
    // multiply-accumulation but still work on all the output elements

    // In register tiled mode, by default each threading block handles an output
    // tile of size (Tm, Tn), produced by A tile in shmem of size (Tm, Tk), and
    // B tile in register of size (Tk, Tn). The threading block size is Tn where
    // data reuse is min(Tm, Tn) and each thread needs to store Tk elements,
    // i.e., ([0:Tk), idx_thread), in registers. Each thread is in charge of
    // ([0:Tn), idx_thread) in the output tile. If the thread number increases,
    // each thread could store a fraction of elements in the registers, and
    // corresponding do a fraction of the multiplication-accumulation. Still,
    // each thread's update will span the whole column it belongs to in the
    // output tile.
    constexpr int COARSEN_DIVISOR_FACTOR_STORE_C =
        (SHMEM_BLOCK_SIZE_X * SHMEM_BLOCK_SIZE_Y) /
        (THREADING_BLOCK_SIZE_X * THREADING_BLOCK_SIZE_Y);
    static_assert(SHMEM_BLOCK_SIZE_Y % THREADING_BLOCK_SIZE_Y == 0, "");
    static_assert(SHMEM_BLOCK_SIZE_X % THREADING_BLOCK_SIZE_X == 0, "");

    constexpr int COARSEN_DIVISOR_FACTOR_LOAD_A =
        SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_Y / THREADING_BLOCK_SIZE_X /
        THREADING_BLOCK_SIZE_Y;
    static_assert((SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_Y) %
                          (THREADING_BLOCK_SIZE_X * THREADING_BLOCK_SIZE_Y) ==
                      0,
                  "");

    constexpr int COARSEN_DIVISOR_FACTOR_LOAD_B =
        (RIGHT_REG_TILED_FLAG && COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT)
            ? SHMEM_BLOCK_SIZE_K
            : (SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_X /
               THREADING_BLOCK_SIZE_X / THREADING_BLOCK_SIZE_Y);
    static_assert((SHMEM_BLOCK_SIZE_K * SHMEM_BLOCK_SIZE_X) %
                          (THREADING_BLOCK_SIZE_X * THREADING_BLOCK_SIZE_Y) ==
                      0,
                  "");

    // Block row and column
    int idx_head = blockIdx.z % num_heads;

    if constexpr (OuterProductFlag) {
      CONSTEXPR_TRUE_CLAUSE_STATIC_ASSERT(OuterProductFlag, AtomicUpdateFlag,
                                          "");
    } else {
      assert((gridDim.z == num_heads));
    }

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
      blockRowLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE_Y);
      blockRowLoopInc = strideNumBlocksAlongRow;
    }

    constexpr int NUM_MUL_ACC =
        COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT
            ? SHMEM_BLOCK_SIZE_K
            : (SHMEM_BLOCK_SIZE_K * COARSEN_DIVISOR_FACTOR_STORE_C /
               SHMEM_BLOCK_SIZE_Y);
    constexpr int NUM_OUTPUT_PER_THREAD = COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT
                                              ? COARSEN_DIVISOR_FACTOR_STORE_C
                                              : SHMEM_BLOCK_SIZE_Y;

    for (int blockRow = blockRowLoopBeg; blockRow < blockRowLoopEnd;
         blockRow += blockRowLoopInc) {
      // NB: blockTask == blockIdx.x / ceil_div( num_B_cols, BLOCK_SIZE)

      // Each thread block computes one sub-matrix Csub of C
      // float* Csub = &C[blockRow * BLOCK_SIZE * num_B_cols + blockFeat *
      // BLOCK_SIZE]; Each thread computes one element of Csub by accumulating
      // results into Cvalue
      // Thread row and column within Csub
      float Cvalue[NUM_OUTPUT_PER_THREAD] = {};

      int thIdx = threadIdx.y * THREADING_BLOCK_SIZE_X + threadIdx.x;
      int thIdxRow_initial_BC = threadIdx.y;
      int thIdxFeat_initial_BC = threadIdx.x;
      if constexpr (THREADING_BLOCK_SIZE_X != SHMEM_BLOCK_SIZE_X) {
        // redo the thread indexing
        thIdxRow_initial_BC = thIdx / SHMEM_BLOCK_SIZE_X;
        thIdxFeat_initial_BC = thIdx % SHMEM_BLOCK_SIZE_X;
      }

      int thIdxRow_initial_A = thIdx / SHMEM_BLOCK_SIZE_K;
      int thIdxFeat_initial_A = thIdx % SHMEM_BLOCK_SIZE_K;

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
        mLoopEnd = ceil_div<>(numARows, (int64_t)SHMEM_BLOCK_SIZE_Y);
        mLoopInc = strideNumBlocksAlongRow;
      } else {
        int InnerProductPartitionIdx = blockIdx.z / num_heads;
        int NumInnerProductionPartitions = blockDim.z / num_heads;
        mLoopBeg = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE_K) *
                   InnerProductPartitionIdx / NumInnerProductionPartitions;
        mLoopEnd = ceil_div<Idx>(num_A_cols, SHMEM_BLOCK_SIZE_K) *
                   (InnerProductPartitionIdx + 1) /
                   NumInnerProductionPartitions;
        mLoopInc = 1;
      }
      for (int m = mLoopBeg; m < mLoopEnd; m += mLoopInc) {
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[SHMEM_BLOCK_SIZE_Y][SHMEM_BLOCK_SIZE_K];
        __shared__ float Bs[RIGHT_REG_TILED_FLAG ? 1 : SHMEM_BLOCK_SIZE_K]
                           [RIGHT_REG_TILED_FLAG ? 1 : SHMEM_BLOCK_SIZE_X];
        float Bs_reg[RIGHT_REG_TILED_FLAG
                         ? (COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT
                                ? SHMEM_BLOCK_SIZE_K
                                : SHMEM_BLOCK_SIZE_K / THREADING_BLOCK_SIZE_Y)
                         : 1];

        // Get sub-matrix Bsub of B
        // Load Asub and Bsub from device memory to
        // shared memory Each thread loads one element of each sub-matrix

        // NB: load_A initial should be based on thIdxRow_initial_A,
        // thIdxFeat_initial_A which uses SHMEM_K as the denominator
        if constexpr (OuterProductFlag) {
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_A;
               loadLoopIdx++) {
            // NB: in outer product, m and y are interchanged, and the loading
            // scheme is a transpose in the fly. that is why both thIdxRow and m
            // need to be used during indexing the row
            // int SHMEM_BLOCK_DIM_Y_PER_LOAD_A_OUTER_PRODUCT =
            //     THREADING_BLOCK_SIZE_X * THREADING_BLOCK_SIZE_Y /
            //     SHMEM_BLOCK_SIZE_K;
            int thIdxRow_A_outer_product =
                (threadIdx.y * THREADING_BLOCK_SIZE_X + threadIdx.x +
                 loadLoopIdx * THREADING_BLOCK_SIZE_X *
                     THREADING_BLOCK_SIZE_Y) %
                SHMEM_BLOCK_SIZE_Y;
            int thIdxFeat_A_outer_product =
                (threadIdx.y * THREADING_BLOCK_SIZE_X + threadIdx.x +
                 loadLoopIdx * THREADING_BLOCK_SIZE_X *
                     THREADING_BLOCK_SIZE_Y) /
                SHMEM_BLOCK_SIZE_Y;

            // Get sub-matrix Asub of A
            As[thIdxRow_A_outer_product][thIdxFeat_A_outer_product] =
                (thIdxFeat_A_outer_product +
                         // k is the row dimension instead of the feature
                         // because of the transpose
                         (m)*SHMEM_BLOCK_SIZE_K <  //+ blockRowJobEntryBeg <
                     numARows &&
                 blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow_A_outer_product <
                     num_A_cols)
                    ? GetRowMajorElement<Idx, IdxPtr, AGatherKind, compactKind>(
                          A, A_gather_list, etype_mapper_data, idx_relation,
                          thIdxFeat_A_outer_product + (m)*SHMEM_BLOCK_SIZE_K +
                              blockRowJobEntryBeg,
                          A_num_head_one_flag ? 0 : idx_head,
                          thIdxRow_A_outer_product +
                              blockRow * SHMEM_BLOCK_SIZE_Y,
                          A_num_head_one_flag ? 1 : num_heads, num_A_cols)
                    : 0.0f;
          }
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_B;
               loadLoopIdx++) {
            // NB: KWU: for outerproduct, the m loop variable is on the K
            // dimension
            int thIdxRow = thIdxRow_initial_BC +
                           loadLoopIdx * (SHMEM_BLOCK_SIZE_K /
                                          COARSEN_DIVISOR_FACTOR_LOAD_B);
            static_assert(
                SHMEM_BLOCK_SIZE_K % COARSEN_DIVISOR_FACTOR_LOAD_B == 0, "");
            int thIdxFeat = thIdxFeat_initial_BC;
            float value_to_load =
                ((m)*SHMEM_BLOCK_SIZE_K + thIdxRow <  //+ blockRowJobEntryBeg <
                     numARows &&
                 blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols &&
                 idx_head < num_heads)
                    ? GetRowMajorElement<Idx, IdxPtr, BGatherKind, compactKind>(
                          B, B_gather_list, etype_mapper_data, idx_relation,
                          (m)*SHMEM_BLOCK_SIZE_K + blockRowJobEntryBeg +
                              thIdxRow,
                          B_num_head_one_flag ? 0 : idx_head,
                          blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat,
                          B_num_head_one_flag ? 1 : num_heads, num_B_cols)
                    : 0.0f;
            if constexpr (RIGHT_REG_TILED_FLAG) {
              Bs_reg[thIdxRow] = value_to_load;
            } else {
              Bs[thIdxRow][thIdxFeat] = value_to_load;
            }
          }
        } else {
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_A;
               loadLoopIdx++) {
            int thIdxRow =
                thIdxRow_initial_A + (loadLoopIdx * SHMEM_BLOCK_SIZE_Y) /
                                         COARSEN_DIVISOR_FACTOR_LOAD_A;
            int thIdxFeat =
                thIdxFeat_initial_A + (loadLoopIdx * SHMEM_BLOCK_SIZE_Y) %
                                          COARSEN_DIVISOR_FACTOR_LOAD_A;
            // Get sub-matrix Asub of A
            As[thIdxRow][thIdxFeat] =
                (thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y < numARows &&
                 m * SHMEM_BLOCK_SIZE_K + thIdxFeat < num_A_cols &&
                 idx_head < num_heads)
                    ? GetRowMajorElement<Idx, IdxPtr, AGatherKind, compactKind>(
                          A, A_gather_list, etype_mapper_data, idx_relation,
                          thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y +
                              blockRowJobEntryBeg,
                          A_num_head_one_flag ? 0 : idx_head,
                          thIdxFeat + m * SHMEM_BLOCK_SIZE_K,
                          A_num_head_one_flag ? 1 : num_heads, num_A_cols)
                    : 0.0f;
          }
          for (int loadLoopIdx = 0; loadLoopIdx < COARSEN_DIVISOR_FACTOR_LOAD_B;
               loadLoopIdx++) {
            int thIdxRow = thIdxRow_initial_BC +
                           loadLoopIdx * (SHMEM_BLOCK_SIZE_K /
                                          COARSEN_DIVISOR_FACTOR_LOAD_B);
            static_assert(
                SHMEM_BLOCK_SIZE_K % COARSEN_DIVISOR_FACTOR_LOAD_B == 0, "");
            int thIdxFeat = thIdxFeat_initial_BC;
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
              float value_to_load =
                  (m * SHMEM_BLOCK_SIZE_K + thIdxRow < num_A_cols &&
                   blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols &&
                   idx_head < num_heads)
                      ? B[(B_num_head_one_flag ? 0 : idx_head) * num_A_cols *
                              num_B_cols +
                          (m * SHMEM_BLOCK_SIZE_K + thIdxRow) * num_B_cols +
                          (blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat)]
                      : 0.0f;
              if constexpr (RIGHT_REG_TILED_FLAG) {
                Bs_reg[loadLoopIdx] = value_to_load;
              } else {
                Bs[thIdxRow][thIdxFeat] = value_to_load;
              }
            } else {  // !BWeightInsteadOfFeatureFlag
              float value_to_load =
                  (m * SHMEM_BLOCK_SIZE_K + thIdxRow < num_A_cols &&
                   blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols &&
                   idx_head < num_heads)
                      ? GetRowMajorElement<Idx, IdxPtr, BGatherKind,
                                           compactKind>(
                            B, B_gather_list, etype_mapper_data, idx_relation,
                            m * SHMEM_BLOCK_SIZE_K + thIdxRow,
                            B_num_head_one_flag ? 0 : idx_head,
                            blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat,
                            B_num_head_one_flag ? 1 : num_heads, num_B_cols)
                      : 0.0f;
              if constexpr (RIGHT_REG_TILED_FLAG) {
                Bs_reg[loadLoopIdx] = value_to_load;
              } else {
                Bs[thIdxRow][thIdxFeat] = value_to_load;
              }
            }
          }
        }

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together

        static_assert(!(RIGHT_REG_TILED_FLAG &&
                        (SHMEM_BLOCK_SIZE_X != THREADING_BLOCK_SIZE_X)),
                      "unsupported yet");

        for (int idx_mul_acc = 0; idx_mul_acc < NUM_MUL_ACC; ++idx_mul_acc) {
          for (int idx_output = 0; idx_output < NUM_OUTPUT_PER_THREAD;
               idx_output++) {
            float right_operand;
            if constexpr (RIGHT_REG_TILED_FLAG) {
              right_operand = Bs_reg[idx_mul_acc];
            } else {
              right_operand = Bs[idx_mul_acc][thIdxFeat_initial_BC];
            }

            Cvalue[idx_output] +=
                As[thIdxRow_initial_BC +
                   idx_output * (SHMEM_BLOCK_SIZE_Y / NUM_OUTPUT_PER_THREAD)]
                  [idx_mul_acc] *
                right_operand;
          }
        }
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }

      // Write Csub to device memory
      // Each thread writes one element
      for (int storeLoopIdx = 0; storeLoopIdx < NUM_OUTPUT_PER_THREAD;
           storeLoopIdx++) {
        int thIdxRow =
            thIdxRow_initial_BC +
            storeLoopIdx * (SHMEM_BLOCK_SIZE_Y / NUM_OUTPUT_PER_THREAD);
        static_assert(
            SHMEM_BLOCK_SIZE_Y % NUM_OUTPUT_PER_THREAD == 0,
            "SHMEM_BLOCK_SIZE_Y must be divisible by NUM_OUTPUT_PER_THREAD");

        int thIdxFeat = thIdxFeat_initial_BC;
        if constexpr (OuterProductFlag) {
          // C is weight instead of feature.
          bool WriteCInRangeFlag =
              blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow < num_A_cols &&
              blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols &&
              idx_head < num_heads;
          if constexpr (!AtomicUpdateFlag) {
            CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(
                OuterProductFlag && AtomicUpdateFlag,
                "OuterproductFlag==true case must use atomic update");
          }
          if (WriteCInRangeFlag) {
            // TODO: after non-atomic update is implemented, make sure that
            // static_assert(!(RIGHT_REG_TILED_FLAG && !AtomicUpdateFlag),
            //          "unsupported");
            // NB: offset dependent on whether one-side num_head is 1
            // if (isnan(Cvalue[storeLoopIdx])) {
            //   printf("nan detected (%d %d %d) (%d %d %d) \n", threadIdx.x,
            //   threadIdx.y,
            //          threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
            // }
            atomicAdd(
                &C[(C_num_head_one_flag ? 0 : idx_head) *
                       num_A_cols /*A is transposed in the fly*/ * num_B_cols +
                   (blockRow * SHMEM_BLOCK_SIZE_Y + thIdxRow) * num_B_cols +
                   blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat],
                Cvalue[storeLoopIdx]);
          }
        } else {  //  !OuterProductFlag

          bool WriteCInRangeFlag =
              thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y < numARows &&
              idx_head < num_heads &&
              blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat < num_B_cols;
          if (WriteCInRangeFlag) {
            auto val_locally_accumulated = Cvalue[storeLoopIdx];
            // printf("%f (%d %d %d) (%d %d %d) \n",val_locally_accumulated,
            // threadIdx.x, threadIdx.y,
            //        threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
            // if (isnan(val_locally_accumulated)) {
            //   printf("nan detected (%d %d %d) (%d %d %d) \n", threadIdx.x,
            //   threadIdx.y,
            //          threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
            // }
            auto &ref_to_global_mem_elem =
                GetRowMajorElement<Idx, IdxPtr, CScatterKind, compactKind>(
                    C, C_scatter_list, etype_mapper_data, idx_relation,
                    thIdxRow + blockRow * SHMEM_BLOCK_SIZE_Y +
                        blockRowJobEntryBeg,
                    C_num_head_one_flag ? 0 : idx_head,
                    blockFeat * SHMEM_BLOCK_SIZE_X + thIdxFeat,
                    C_num_head_one_flag ? 1 : num_heads, num_B_cols);
            // TODO: optimize this so that no need to do atomic update when reg
            // tiling is used
            if constexpr (AtomicUpdateFlag ||
                          !COARSEN_OUTPUT_INSTEAD_OF_RIGHT_INPUT) {
              atomicAdd(&ref_to_global_mem_elem, val_locally_accumulated);
            } else {  // !AtomicUpdateFlag
              ref_to_global_mem_elem = val_locally_accumulated;
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

// TODO: KWU: tweak launch_bounds via template variables
// TODO: KWU: add a new reg tile version
// blockIdx.y == ceil_div (num_edges, BLOCK_SIZE)
// FIXME: nan here
template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void MY_SGEMM_LAUNCH_BOUNDS HET_RGNNFeatPerEdgeFwProp(
    float *node_feat_input, float *weight, float *node_feat_per_edge,
    IdxPtr A_col_row_idx_gather_list, IdxPtr A_rel_ptr,
    IdxPtr C_eid_scatter_list, Idx input_dim, Idx output_per_head_dim,
    int num_heads, int *accum_num_blocks_per_relation, Idx num_relations) {
  // (input, weight, output) are 1, NH, NH or NH, NH, NH depending on whether
  // A_num_head_one_flag is true. NH is num_heads
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<
      RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
      WORK_BLOCK_SIZE_K, false, MySGEMMGatherKind::Basic,
      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Basic, false, Idx, IdxPtr,
      MySGEMMNumHeadKind::AssertANumIsOne, CompactAsOfNodeKind::Disabled>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
          node_feat_per_edge, A_col_row_idx_gather_list, nullptr,
          C_eid_scatter_list, {}, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

// for HGT nodewise linear layers
template <bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
          int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X,
          int WORK_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_K, typename Idx,
          typename IdxPtr>
__global__ void NO_SCATTER_GATHER_LAUNCH_BOUNDS
HET_RGNNMatmulNoScatterGatherListFwOrBckProp(
    float *node_feat_input, float *weights, float *linear_projected_node_feat,
    IdxPtr ntype_ptrs, int *accum_num_blocks_per_ntype, Idx num_ntypes,
    Idx input_dim, Idx output_dim) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_ntype = binary_search<int, int *>(
      num_ntypes, accum_num_blocks_per_ntype, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                      WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, false,
                      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Disabled,
                      // This function is kind of exceptional: we need to assert
                      // all the three num_heads are 1
                      MySGEMMGatherKind::Disabled, false, Idx, IdxPtr,
                      MySGEMMNumHeadKind::AssertAllAreOnes,
                      CompactAsOfNodeKind::Disabled>::
      execute_function(node_feat_input, weights, linear_projected_node_feat,
                       nullptr, nullptr, nullptr, {}, 0,
                       ntype_ptrs[idx_ntype + 1] - ntype_ptrs[idx_ntype],
                       accum_num_blocks_per_ntype[idx_ntype],
                       (accum_num_blocks_per_ntype[idx_ntype + 1] -
                        accum_num_blocks_per_ntype[idx_ntype]),
                       ntype_ptrs[idx_ntype], input_dim, output_dim, 1);
}

// for HGT nodewise linear layers
template <bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
          int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X,
          int WORK_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_K, typename Idx,
          typename IdxPtr>
__global__ void HET_RGNNDeltaWeightNoScatterGatherListBckProp(
    float *node_feat_input, float *delta_feat, float *delta_weight,
    IdxPtr ntype_ptrs, Idx A_input_dim, Idx B_delta_output_dim,
    int *accum_num_blocks_per_ntype, Idx num_ntypes) {
  Idx idx_block_assignment = blockIdx.z;
  Idx idx_ntype = binary_search<int, int *>(
      num_ntypes, accum_num_blocks_per_ntype, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                      WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, true,
                      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Disabled,
                      MySGEMMGatherKind::Disabled, true, Idx,
                      // This function is kind of exceptional: we need to assert
                      // all the three num_heads are 1
                      IdxPtr, MySGEMMNumHeadKind::AssertAllAreOnes,
                      CompactAsOfNodeKind::Disabled>::
      execute_function(
          node_feat_input, delta_feat,
          &delta_weight[idx_ntype * A_input_dim * B_delta_output_dim], nullptr,
          nullptr, nullptr, {}, idx_ntype,
          ntype_ptrs[idx_ntype + 1] - ntype_ptrs[idx_ntype],
          accum_num_blocks_per_ntype[idx_ntype],
          (accum_num_blocks_per_ntype[idx_ntype + 1] -
           accum_num_blocks_per_ntype[idx_ntype]),
          ntype_ptrs[idx_ntype], A_input_dim, B_delta_output_dim, 1);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNFeatPerEdgeFwPropACGatherScatterListIdentical(
    float *node_feat_input, float *weight, float *node_feat_per_edge,
    IdxPtr A_rel_ptr, IdxPtr AC_eid_gather_scatter_list, Idx input_dim,
    Idx output_per_head_dim, int num_heads, int *accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                      WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, false,
                      MySGEMMGatherKind::Basic, MySGEMMGatherKind::Disabled,
                      MySGEMMGatherKind::Basic, false, Idx,
                      // (input, weight, output) are 1, NH, NH or NH, NH, NH
                      // depending on whether A_num_head_one_flag is true. NH is
                      // num_heads
                      IdxPtr, MySGEMMNumHeadKind::AssertANumIsOne,
                      CompactAsOfNodeKind::Disabled>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
          node_feat_per_edge, AC_eid_gather_scatter_list, nullptr,
          AC_eid_gather_scatter_list, {}, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], input_dim, output_per_head_dim, num_heads);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void MY_SGEMM_LAUNCH_BOUNDS HET_RGNNFeatCompactFwProp(
    float *node_feat_input, float *weight, float *node_feat_per_edge,
    const ETypeMapperData<Idx, CompactAsOfNodeKind::Enabled> etype_mapper_data,
    Idx input_dim, Idx output_per_head_dim, int num_heads,
    int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<
      RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
      WORK_BLOCK_SIZE_K, false, MySGEMMGatherKind::Basic,
      MySGEMMGatherKind::Disabled,
      // NB: no need to scatter C
      // (input, weight, output) are 1, NH, NH or NH, NH, NH depending
      // on whether A_num_head_one_flag is true. NH is num_heads
      MySGEMMGatherKind::Disabled, false, Idx, IdxPtr,
      MySGEMMNumHeadKind::AssertANumIsOne, CompactAsOfNodeKind::Enabled>::
      execute_function(
          node_feat_input,
          &weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                  input_dim * output_per_head_dim],
          node_feat_per_edge,
          etype_mapper_data.unique_srcs_and_dests_node_indices, nullptr,
          etype_mapper_data.unique_srcs_and_dests_node_indices,
          etype_mapper_data, idx_relation,
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation + 1] -
              etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
          input_dim, output_per_head_dim, num_heads);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void MY_SGEMM_LAUNCH_BOUNDS HET_RGNNDeltaNodeFeatInputBckProp(
    float *delta_feat_per_edge, float *weight_transposed,
    float *delta_node_input, IdxPtr A_eid_gather_list, IdxPtr A_rel_ptr,
    IdxPtr C_col_row_idx_scatter_list, Idx delta_output_per_head_dim,
    Idx delta_input_dim, int num_heads, int *accum_num_blocks_per_relation,
    Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y,
                      // (delta_feat, weight, delta_input) are NH, NH, 1 or NH,
                      // NH, NH depending on whether C_num_head_one_flag is
                      // true. NH is num_heads
                      WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K,
                      false, MySGEMMGatherKind::Basic,
                      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Basic,
                      true, Idx, IdxPtr, MySGEMMNumHeadKind::AssertCNumIsOne,
                      CompactAsOfNodeKind::Disabled>::
      execute_function(
          delta_feat_per_edge,
          &weight_transposed[idx_relation *
                             (InputNumHeadOneFlag ? num_heads : 1) *
                             delta_input_dim * delta_output_per_head_dim],
          delta_node_input, A_eid_gather_list, nullptr,
          C_col_row_idx_scatter_list, {}, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], delta_output_per_head_dim, delta_input_dim,
          num_heads);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightBckProp(
    float *node_feat_input, float *delta_feat_per_edge, float *delta_weight,
    IdxPtr A_col_row_idx_gather_list, IdxPtr A_rel_ptr,
    IdxPtr B_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, int num_heads,
    int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<
      RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
      WORK_BLOCK_SIZE_K, true, MySGEMMGatherKind::Basic,
      MySGEMMGatherKind::Basic, MySGEMMGatherKind::Disabled, true, Idx, IdxPtr,
      MySGEMMNumHeadKind::AssertANumIsOne, CompactAsOfNodeKind::Disabled>::
      execute_function(
          node_feat_input, delta_feat_per_edge,
          &delta_weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          A_col_row_idx_gather_list, B_eid_gather_list, nullptr, {},
          idx_relation, A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], A_delta_input_dim,
          B_delta_output_per_head_dim, num_heads);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightBckPropACGatherScatterListIdentical(
    float *node_feat_input, float *delta_feat_per_edge, float *delta_weight,
    IdxPtr A_rel_ptr, IdxPtr AB_eid_gather_list, Idx A_delta_input_dim,
    Idx B_delta_output_per_head_dim, int num_heads,
    int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<
      RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
      WORK_BLOCK_SIZE_K, true, MySGEMMGatherKind::Basic,
      MySGEMMGatherKind::Basic, MySGEMMGatherKind::Disabled, true, Idx, IdxPtr,
      // (delta_feat, weight, delta_input) are NH, NH, 1 or NH,
      // NH, NH depending on whether C_num_head_one_flag is
      // true. NH is num_heads
      MySGEMMNumHeadKind::AssertCNumIsOne, CompactAsOfNodeKind::Disabled>::
      execute_function(
          node_feat_input, delta_feat_per_edge,
          &delta_weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          AB_eid_gather_list, AB_eid_gather_list, nullptr, {}, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], A_delta_input_dim,
          B_delta_output_per_head_dim, num_heads);
}

// blockDim.y == ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaWeightCompactBckProp(
    float *delta_feat_compact, float *feat_input, float *delta_weight,
    const ETypeMapperData<Idx, CompactAsOfNodeKind::Enabled> etype_mapper_data,
    Idx num_edges, Idx A_delta_input_dim, Idx B_delta_output_per_head_dim,
    int num_heads, int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.z / num_heads;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_X,
                      WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K, true,
                      MySGEMMGatherKind::Basic, MySGEMMGatherKind::Disabled,
                      MySGEMMGatherKind::Disabled, true, Idx,
                      // (delta_feat, weight, delta_input) are NH, NH, 1 or NH,
                      // NH, NH depending on whether C_num_head_one_flag is
                      // true. NH is num_heads
                      IdxPtr, MySGEMMNumHeadKind::AssertCNumIsOne,
                      CompactAsOfNodeKind::Enabled>::
      execute_function(
          feat_input, delta_feat_compact,
          &delta_weight[idx_relation * (InputNumHeadOneFlag ? num_heads : 1) *
                        B_delta_output_per_head_dim * A_delta_input_dim],
          etype_mapper_data.unique_srcs_and_dests_node_indices,
          etype_mapper_data.unique_srcs_and_dests_node_indices, nullptr,
          etype_mapper_data, idx_relation,
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation + 1] -
              etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
          A_delta_input_dim, B_delta_output_per_head_dim, num_heads);
}

template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaNodeFeatInputBckPropACGatherScatterListIdentical(
    float *delta_feat_per_edge, float *weight_transposed,
    float *delta_node_input, IdxPtr A_C_eid_gather_scatter_list,
    IdxPtr A_rel_ptr, Idx delta_output_per_head_dim, Idx delta_input_dim,
    int num_heads, int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y,
                      // (delta_feat, weight, delta_input) are NH, NH, 1 or NH,
                      // NH, NH depending on whether C_num_head_one_flag is
                      // true. NH is num_heads
                      WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K,
                      false, MySGEMMGatherKind::Basic,
                      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Basic,
                      true, Idx, IdxPtr, MySGEMMNumHeadKind::AssertCNumIsOne,
                      CompactAsOfNodeKind::Disabled>::
      execute_function(
          delta_feat_per_edge,
          &weight_transposed[idx_relation *
                             (InputNumHeadOneFlag ? num_heads : 1) *
                             delta_input_dim * delta_output_per_head_dim],
          delta_node_input, A_C_eid_gather_scatter_list, nullptr,
          A_C_eid_gather_scatter_list, {}, idx_relation,
          A_rel_ptr[idx_relation + 1] - A_rel_ptr[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          A_rel_ptr[idx_relation], delta_output_per_head_dim, delta_input_dim,
          num_heads);
}

// blockDim.y == ceil_div(A_col_row_idx_gather_list.size(), BLOCK_SIZE)
template <
    bool RIGHT_REG_TILED_FLAG, int THREADING_BLOCK_SIZE_X,
    int THREADING_BLOCK_SIZE_Y, int WORK_BLOCK_SIZE_X, int WORK_BLOCK_SIZE_Y,
    int WORK_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
    bool InputNumHeadOneFlag /*whether (delta_)input_feat is single-headed*/>
__global__ void HET_RGNNDeltaNodeFeatInputCompactBckProp(
    float *delta_feat_compact, float *weight_transpose,
    float *delta_node_feat_input,
    const ETypeMapperData<Idx, CompactAsOfNodeKind::Enabled> etype_mapper_data,
    Idx num_edges, Idx delta_output_per_head_dim, Idx delta_input_dim,
    int num_heads, int *accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.y;
  Idx idx_relation = binary_search<int, int *>(
      num_relations, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<RIGHT_REG_TILED_FLAG, false, THREADING_BLOCK_SIZE_X,
                      THREADING_BLOCK_SIZE_Y,
                      // (delta_feat, weight, delta_input) are NH, NH, 1 or NH,
                      // NH, NH depending on whether C_num_head_one_flag is
                      // true. NH is num_heads
                      WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y, WORK_BLOCK_SIZE_K,
                      false, MySGEMMGatherKind::Disabled,
                      MySGEMMGatherKind::Disabled, MySGEMMGatherKind::Basic,
                      true, Idx, IdxPtr, MySGEMMNumHeadKind::AssertCNumIsOne,
                      CompactAsOfNodeKind::Enabled>::
      execute_function(
          delta_feat_compact,
          &weight_transpose[idx_relation *
                            (InputNumHeadOneFlag ? num_heads : 1) *
                            delta_output_per_head_dim * delta_input_dim],
          delta_node_feat_input,
          etype_mapper_data.unique_srcs_and_dests_node_indices, nullptr,
          etype_mapper_data.unique_srcs_and_dests_node_indices,
          etype_mapper_data, idx_relation,
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation + 1] -
              etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
          accum_num_blocks_per_relation[idx_relation],
          (accum_num_blocks_per_relation[idx_relation + 1] -
           accum_num_blocks_per_relation[idx_relation]),
          etype_mapper_data.unique_srcs_and_dests_rel_ptrs[idx_relation],
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
