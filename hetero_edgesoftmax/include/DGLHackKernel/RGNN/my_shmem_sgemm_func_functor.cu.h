#pragma once
#include <cuda_runtime.h>
// #include "cuda.h"
#include "utils.cu.h"

enum class MySGEMMNumHeadKind {
  NoAssert = 0,
  AssertANumIsOne,
  AssertCNumIsOne,
  AssertAllAreOnes
};

enum class MySGEMMGatherKind {
  Disabled = 0,
  Basic,
  TwoOrderBinarySearch,
  TwoOrderDirectIndexing
};

template <MySGEMMGatherKind kind, typename Idx, typename IdxPtr>
__device__ __forceinline__ float &GetRowMajorElementBasic(
    float *matrix_data, IdxPtr gather_list, int num_heads,
    Idx feat_dim_per_head, Idx row, Idx idx_head, Idx idx_feat) {
  if constexpr (kind != MySGEMMGatherKind::Disabled) {
    return matrix_data[idx_head * feat_dim_per_head +
                       gather_list[row] * num_heads * feat_dim_per_head +
                       idx_feat];
  } else {
    return matrix_data[idx_head * feat_dim_per_head +
                       row * num_heads * feat_dim_per_head + idx_feat];
  }
}

template <typename Idx, typename IdxPtr>
__device__ __forceinline__ float &GetRowMajorElementAdvancedBinarySearch(
    float *matrix_data, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx idx_node,
    Idx idx_head, Idx idx_feat, int num_heads, Idx feat_dim_per_head) {
  Idx offset = find_relational_compact_as_of_node_index(
      idx_relation, idx_node, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices);
  return GetRowMajorElementBasic<MySGEMMGatherKind::Disabled, Idx, IdxPtr>(
      matrix_data, nullptr, num_heads, feat_dim_per_head, offset, idx_head,
      idx_feat);
}

template <typename Idx, typename IdxPtr, MySGEMMGatherKind kind,
          CompactAsOfNodeKind compactKind>
__device__ __forceinline__ float &GetRowMajorElement(
    float *matrix_data, IdxPtr gather_scatter_list,
    ETypeMapperData<Idx, compactKind> etype_mapper_data, Idx idx_relation,
    Idx idx_row, Idx idx_head, Idx idx_feat, int num_heads,
    Idx feat_dim_per_head) {
  if constexpr (kind == MySGEMMGatherKind::TwoOrderBinarySearch ||
                kind == MySGEMMGatherKind::TwoOrderDirectIndexing) {
    Idx idx_node = gather_scatter_list[idx_row];
    if constexpr (kind == MySGEMMGatherKind::TwoOrderBinarySearch) {
      return GetRowMajorElementAdvancedBinarySearch<Idx, IdxPtr>(
          matrix_data, etype_mapper_data.unique_srcs_and_dests_rel_ptr,
          etype_mapper_data.unique_srcs_and_dests_node_indices, idx_relation,
          idx_node, idx_head, idx_feat, num_heads, feat_dim_per_head);
    } else {
      // return GetRowMajorElementBasic<false, Idx, IdxPtr>(
      //    matrix_data, nullptr, num_heads, feat_dim_per_head, idx_node,
      //    idx_head, idx_feat);
      CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
          kind == MySGEMMGatherKind::TwoOrderDirectIndexing, "not implemented");
    }
  } else {
    return GetRowMajorElementBasic<kind, Idx, IdxPtr>(
        matrix_data, gather_scatter_list, num_heads, feat_dim_per_head, idx_row,
        idx_head, idx_feat);
  }
}

template <bool DOUBLE_BUFFER_FLAG, int THREADING_BLOCK_SIZE_X,
          int THREADING_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_X,
          int SHMEM_BLOCK_SIZE_Y, int SHMEM_BLOCK_SIZE_K, bool OuterProductFlag,
          MySGEMMGatherKind AGatherKind, MySGEMMGatherKind BGatherKind,
          MySGEMMGatherKind CScatterKind, bool AtomicUpdateFlag, typename Idx,
          typename IdxPtr, MySGEMMNumHeadKind numHeadKind,
          CompactAsOfNodeKind compactKind>
class _basic_MatMulKernel {
  __device__ __forceinline__ static void execute_function(
      float *A, float *B, float *C, IdxPtr A_gather_list, IdxPtr B_gather_list,
      IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
      IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx numARows,
      Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    assert(0 && "not implemented");
  }
};

// the double buffer version
template <int THREADING_BLOCK_SIZE_X, int THREADING_BLOCK_SIZE_Y,
          int SHMEM_BLOCK_SIZE_X, int SHMEM_BLOCK_SIZE_Y,
          int SHMEM_BLOCK_SIZE_K, bool OuterProductFlag,
          MySGEMMGatherKind AGatherKind, MySGEMMGatherKind BGatherKind,
          MySGEMMGatherKind CScatterKind, bool AtomicUpdateFlag, typename Idx,
          typename IdxPtr, MySGEMMNumHeadKind numHeadKind,
          CompactAsOfNodeKind compactKind>
class _basic_MatMulKernel<true, THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
                          SHMEM_BLOCK_SIZE_X, SHMEM_BLOCK_SIZE_Y,
                          SHMEM_BLOCK_SIZE_K, OuterProductFlag, AGatherKind,
                          BGatherKind, CScatterKind, AtomicUpdateFlag, Idx,
                          IdxPtr, numHeadKind, compactKind> {
 public:
  __device__ __forceinline__ static void execute_function(
      float *A, float *B, float *C, IdxPtr A_gather_list, IdxPtr B_gather_list,
      IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
      IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx numARows,
      Idx blockIdxAlongRowBeg, Idx strideNumBlocksAlongRow,
      Idx blockRowJobEntryBeg, Idx num_A_cols, Idx num_B_cols, int num_heads) {
    assert(0 && "not implemented");
  }
};
