#pragma once
#include <cuda_runtime.h>
// #include "cuda.h"
#include "utils.cu.h"

// DoInnerProductSwitch 0: no inner product, 1: do inner product,
// 2: do inner product and do not output C
enum class MySGEMMInnerProductKind { Disabled, Enabled, EnabledAndSkipOutputC };

template <bool DOUBLE_BUFFER_FLAG, int THREAD_BLOCK_DIM_X,
          int THREAD_BLOCK_DIM_Y, int SHMEM_BLOCK_SIZE_X,
          int SHMEM_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_K, typename Idx,
          typename IdxPtr, bool HGT_INSTEAD_OF_RGCN_FLAG, bool OuterProductFlag,
          MySGEMMInnerProductKind DoInnerProductSwitch,
          bool InnerProductGatherListNodeInsteadOfEdge, bool NoEdgeNormFlag,
          bool AtomicUpdateFlag>
class _simplified_basic_MatMulKernel {
 public:
  __device__ __forceinline__ static void execute_function(
      float* A, float* B, float* C, float* edge_norm, float* inner_product,
      float* input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
      IdxPtr separate_coo_col_idx, IdxPtr separate_coo_eids, Idx idx_relation,
      Idx numARows, Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    assert(0 && "not implemented");
    // CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(DOUBLE_BUFFER_FLAG&&DOUBLE_BUFFER_FLAG,
    // "only partial specialized version should be called");
  }
};

template <int THREAD_BLOCK_DIM_X, int THREAD_BLOCK_DIM_Y,
          int SHMEM_BLOCK_SIZE_X, int SHMEM_BLOCK_SIZE_Y,
          int SHMEM_BLOCK_SIZE_K, typename Idx, typename IdxPtr,
          bool HGT_INSTEAD_OF_RGCN_FLAG, bool OuterProductFlag,
          MySGEMMInnerProductKind DoInnerProductSwitch,
          bool InnerProductGatherListNodeInsteadOfEdge, bool NoEdgeNormFlag,
          bool AtomicUpdateFlag>
class _simplified_basic_MatMulKernel<
    true, THREAD_BLOCK_DIM_X, THREAD_BLOCK_DIM_Y, SHMEM_BLOCK_SIZE_X,
    SHMEM_BLOCK_SIZE_Y, SHMEM_BLOCK_SIZE_K, Idx, IdxPtr,
    HGT_INSTEAD_OF_RGCN_FLAG, OuterProductFlag, DoInnerProductSwitch,
    InnerProductGatherListNodeInsteadOfEdge, NoEdgeNormFlag, AtomicUpdateFlag> {
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
      float* A, float* B, float* C, float* edge_norm, float* inner_product,
      float* input_node_feat_for_inner_product, IdxPtr separate_coo_row_idx,
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
    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
        HGT_INSTEAD_OF_RGCN_FLAG && !HGT_INSTEAD_OF_RGCN_FLAG,
        "double buffer version is obsolete and should be updated to align with "
        "the single buffer version");
  }
};
