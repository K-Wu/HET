#pragma once
#include <cuda_runtime.h>
#include "cuda.h"
#include "utils.cu.h"

template <bool GatherFlag, typename Idx, typename IdxPtr>
__device__ __forceinline__ float& GetRowMajorElementBasic(
    float* matrix_data, IdxPtr gather_list, int num_heads,
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
    Idx idx_head, Idx idx_feat, int num_heads, Idx feat_dim_per_head) {
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
    Idx idx_head, Idx idx_feat, int num_heads, Idx feat_dim_per_head) {
  if constexpr (AdvancedGatherScatterFlag) {
    Idx idx_node = gather_scatter_list[idx_row];
    if constexpr (GatherScatterFlag) {
      return GetRowMajorElementAdvanced<Idx, IdxPtr>(
          matrix_data, unique_srcs_and_dests_rel_ptr,
          unique_srcs_and_dests_node_indices, idx_relation, idx_node, idx_head,
          idx_feat, num_heads, feat_dim_per_head);
    } else {
      // return GetRowMajorElementBasic<false, Idx, IdxPtr>(
      //    matrix_data, nullptr, num_heads, feat_dim_per_head, idx_node,
      //    idx_head, idx_feat);
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          AdvancedGatherScatterFlag && !GatherScatterFlag, "");
    }

  } else {
    return GetRowMajorElementBasic<GatherScatterFlag, Idx, IdxPtr>(
        matrix_data, gather_scatter_list, num_heads, feat_dim_per_head, idx_row,
        idx_head, idx_feat);
  }
}

template <bool DOUBLE_BUFFER_FLAG, bool COARSEN_FACTOR_2_FLAG_X,
          bool COARSEN_FACTOR_2_FLAG_Y, int SHMEM_BLOCK_SIZE,
          bool OuterProductFlag, bool GatherAFlag, bool AdvancedGatherAFlag,
          bool GatherBFlag, bool AdvancedGatherBFlag, bool ScatterCFlag,
          bool AdvancedScatterCFlag, bool AtomicUpdateFlag, typename Idx,
          typename IdxPtr, bool A_num_head_one_flag, bool B_num_head_one_flag,
          bool C_num_head_one_flag>
class _basic_MatMulKernel {
  __device__ __forceinline__ static void execute_function(
      float* A, float* B, float* C, IdxPtr A_gather_list, IdxPtr B_gather_list,
      IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
      IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx numARows,
      Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    assert(0 && "not implemented");
  }
};

// the double buffer version
template <bool COARSEN_FACTOR_2_FLAG_X, bool COARSEN_FACTOR_2_FLAG_Y,
          int SHMEM_BLOCK_SIZE, bool OuterProductFlag, bool GatherAFlag,
          bool AdvancedGatherAFlag, bool GatherBFlag, bool AdvancedGatherBFlag,
          bool ScatterCFlag, bool AdvancedScatterCFlag, bool AtomicUpdateFlag,
          typename Idx, typename IdxPtr, bool A_num_head_one_flag,
          bool B_num_head_one_flag, bool C_num_head_one_flag>
class _basic_MatMulKernel<
    true, COARSEN_FACTOR_2_FLAG_X, COARSEN_FACTOR_2_FLAG_Y, SHMEM_BLOCK_SIZE,
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
    assert(0 && "not implemented");
  }
};
