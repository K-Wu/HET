#pragma once
#include <cuda_runtime.h>
#include "cuda.h"
#include "utils.cu.h"

template <bool GatherFlag, typename Idx, typename IdxPtr>
__device__ __forceinline__ float& GetRowMajorElementBasic(
    float* matrix_data, IdxPtr gather_list, Idx num_cols, Idx feat_dim_per_head,
    Idx row, Idx idx_head, Idx idx_feat) {
  if constexpr (GatherFlag) {
    return matrix_data[(idx_head * feat_dim_per_head) +
                       (gather_list[row]) * num_cols + (idx_feat)];
  } else {
    return matrix_data[(idx_head * feat_dim_per_head) + (row)*num_cols +
                       (idx_feat)];
  }
}

template <typename Idx, typename IdxPtr>
__device__ __forceinline__ float& GetRowMajorElementAdvanced(
    float* matrix_data, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx idx_node,
    Idx idx_head, Idx idx_feat, Idx num_cols, Idx feat_dim_per_head) {
  Idx offset = find_relational_compact_as_of_node_index(
      idx_relation, idx_node, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices);
  return GetRowMajorElementBasic<false, Idx, IdxPtr>(
      matrix_data, nullptr, num_cols, feat_dim_per_head, offset, idx_head,
      idx_feat);
}

template <typename Idx, typename IdxPtr, bool GatherScatterFlag,
          bool AdvancedGatherScatterFlag>
__device__ __forceinline__ float& GetRowMajorElement(
    float* matrix_data, IdxPtr gather_scatter_list,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx idx_row,
    Idx idx_head, Idx idx_feat, Idx num_cols, Idx feat_dim_per_head) {
  if constexpr (GatherScatterFlag && AdvancedGatherScatterFlag) {
    Idx idx_node = gather_scatter_list[idx_row];
    return GetRowMajorElementAdvanced<Idx, IdxPtr>(
        matrix_data, unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices, idx_relation, idx_node, idx_head,
        idx_feat, num_cols, feat_dim_per_head);
  } else {
    return GetRowMajorElementBasic<true, Idx, IdxPtr>(
        matrix_data, gather_scatter_list, num_cols, feat_dim_per_head, idx_row,
        idx_head, idx_feat);
  }
}

template <typename Idx>
__device__ __host__ __forceinline__ Idx ceil_div(Idx a, Idx b) {
  return (a + b - 1) / b;
}

// code from
// http://www.shodor.org/media/content//petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf
//@@ Example of grid and block configuration
//	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
// dim3 dimGrid(num_assignment_units * B.width / dimBlock.x, A.height /
// dimBlock.y, num_heads * num_partitions_along_Acol_Brow);
// MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
// C = A * B (all are row-major matrices)
template <int BLOCK_SIZE, bool OuterProductFlag, bool GatherAFlag,
          bool AdvancedGatherAFlag, bool GatherBFlag, bool AdvancedGatherBFlag,
          bool ScatterCFlag, bool AdvancedScatterCFlag, typename Idx,
          typename IdxPtr>
__device__ __forceinline__ void _basic_MatMulKernel(
    float* A, float* B, float* C, IdxPtr A_gather_list, IdxPtr B_gather_list,
    IdxPtr C_scatter_list, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx idx_relation, Idx num_A_rows,
    Idx num_A_cols, Idx num_B_cols, Idx num_heads) {
  constexpr bool AInFlyTransposeFlag = OuterProductFlag;
  Idx A_feat_dim_per_head = num_A_cols / num_heads;
  Idx B_feat_dim_per_head = num_B_cols / num_heads;

  // Block row and column
  Idx idx_head = blockIdx.z % num_heads;
  Idx InnerProductPartitionIdx = blockIdx.z / num_heads;
  Idx NumInnerProductionPartitions = blockDim.z / num_heads;
  if constexpr (!OuterProductFlag) {
    assert((blockDim.z == num_heads));
  }
  Idx blockRow = blockIdx.y;
  Idx blockCol = blockIdx.x % ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  // NB: blockTask == blockIdx.x / ceil_div( num_B_cols, BLOCK_SIZE)

  // Each thread block computes one sub-matrix Csub of C
  float* Csub = &C[blockRow * BLOCK_SIZE * num_B_cols + blockCol * BLOCK_SIZE];
  // Each thread computes one element of Csub
  // by accumulating results into Cvalue
  float Cvalue = 0.0f;
  // Thread row and column within Csub
  Idx row = threadIdx.y;
  Idx col = threadIdx.x;
  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (Idx m = ceil_div<Idx>(num_A_cols, BLOCK_SIZE) *
               InnerProductPartitionIdx / NumInnerProductionPartitions;
       m < ceil_div<Idx>(num_A_cols, BLOCK_SIZE) *
               (InnerProductPartitionIdx + 1) / NumInnerProductionPartitions;
       ++m) {
    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Get sub-matrix Bsub of B
    float* Bsub = &B[m * num_B_cols * BLOCK_SIZE + blockCol * BLOCK_SIZE];
    float* Asub;
    // Load Asub and Bsub from device memory to shared memory
    // Each thread loads one element of each sub-matrix
    if constexpr (AInFlyTransposeFlag) {
      // Get sub-matrix Asub of A
      Asub = &A[m * BLOCK_SIZE * num_A_cols + blockRow * BLOCK_SIZE];
      As[col][row] =
          (row + m * BLOCK_SIZE < num_A_rows &&
           blockRow * BLOCK_SIZE + col + idx_head * A_feat_dim_per_head <
               num_A_cols)
              ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                   AdvancedGatherAFlag>(
                    A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                    unique_srcs_and_dests_node_indices, idx_relation,
                    row + m * BLOCK_SIZE, idx_head, col + blockRow * BLOCK_SIZE,
                    num_A_cols, A_feat_dim_per_head)
              : 0.0f;
    } else {
      // Get sub-matrix Asub of A
      Asub = &A[blockRow * BLOCK_SIZE * num_A_cols + m * BLOCK_SIZE];
      As[row][col] =
          (row + blockRow * BLOCK_SIZE < num_A_rows &&
           m * BLOCK_SIZE + col + idx_head * A_feat_dim_per_head < num_A_cols)
              ? GetRowMajorElement<Idx, IdxPtr, GatherAFlag,
                                   AdvancedGatherAFlag>(
                    A, A_gather_list, unique_srcs_and_dests_rel_ptr,
                    unique_srcs_and_dests_node_indices, idx_relation,
                    row + blockRow * BLOCK_SIZE, idx_head, col + m * BLOCK_SIZE,
                    num_A_cols, A_feat_dim_per_head)
              : 0.0f;
    }

    Bs[row][col] =
        (m * BLOCK_SIZE + row < num_A_cols &&
         blockCol * BLOCK_SIZE + col < num_B_cols)
            ? GetRowMajorElement<Idx, IdxPtr, GatherBFlag, AdvancedGatherBFlag>(
                  B, B_gather_list, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices, idx_relation,
                  m * BLOCK_SIZE + row,
                  blockCol * BLOCK_SIZE + col + idx_head * B_feat_dim_per_head,
                  num_B_cols, B_feat_dim_per_head)
            : 0.0f;
    // Synchronize to make sure the sub-matrices are loaded
    // before starting the computation
    __syncthreads();
    // Multiply Asub and Bsub together
    for (int e = 0; e < BLOCK_SIZE; ++e) Cvalue += As[row][e] * Bs[e][col];
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }
  // Write Csub to device memory
  // Each thread writes one element
  if (blockRow * BLOCK_SIZE + row < num_A_rows &&
      blockCol * BLOCK_SIZE + col < num_B_cols) {
    if constexpr (OuterProductFlag) {
      atomicAdd(
          &GetRowMajorElement<Idx, IdxPtr, ScatterCFlag, AdvancedScatterCFlag>(
              C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
              unique_srcs_and_dests_node_indices, idx_relation,
              blockRow * BLOCK_SIZE + row, blockCol * BLOCK_SIZE + col,
              num_B_cols, B_feat_dim_per_head),
          Cvalue);
    } else {
      GetRowMajorElement<Idx, IdxPtr, ScatterCFlag, AdvancedScatterCFlag>(
          C, C_scatter_list, unique_srcs_and_dests_rel_ptr,
          unique_srcs_and_dests_node_indices, idx_relation,
          blockRow * BLOCK_SIZE + row, blockCol * BLOCK_SIZE + col, num_B_cols,
          B_feat_dim_per_head) = Cvalue;
    }
  }
}

// C = A * B all are row major

// configurations of forward propagation procedures

// feat_per_edge = input * weight. C needs eid scatter list, A needs col|row_idx
// gather list. GatherA = true, ScatterC = true

// feat_compact = input * weight. C needs (eid, idx_relation, unique) -> offset
// map. A needs col|row_idx gather list, GatherA =true, ScatterC = true,
// AdvancedScatterC = true.

// configurations of backward propagation procedures

// delta node_input = delta_feat_per_edge * weightT: A needs gather list eid, C
// needs col|row_idx as scatter list. GatherA = true, ScatterC = true

// delta_weight = feat_input T * delta feat_per_edge: B needs eid as gather
// list, A needs col|row_idx as gather list. GatherA = true, GatherB = true,
// AInFlyTranspose = true

// delta input = delta feat_compact * weight T : A needs (eid, idx_relation,
// unique) -> offset map, GatherA=true, AdvancedGatherA=true

// delta weight =  feat_input T * delta feat_compact: B needs (eid,
// idx_relation, unique) -> offset map, A needs col|row_idx as gather list.
// GatherA=true, GatherB=true, AdvancedGatherB=true, AInFlyTranspose = true

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatPerEdgeFWProp(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr A_col_row_idx_gather_list, IdxPtr C_eid_scatter_list, Idx num_A_rows,
    Idx num_A_cols, Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  int idx_block_assignment = Idx idx_block_assignment =
      blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, false, true, false, false, false, true, false,
                      Idx, IdxPtr>(
      node_feat_input, weight, node_feat_per_edge, A_col_row_idx_gather_list,
      nullptr, C_eid_scatter_list, nullptr, nullptr, idx_relation, num_A_rows,
      num_A_cols, num_B_cols, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNFeatCompactFWProp(
    float* node_feat_input, float* weight, float* node_feat_per_edge,
    IdxPtr A_col_row_idx_gather_list, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_A_rows, Idx num_A_cols,
    Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, false, true, false, false, false, true, true,
                      Idx, IdxPtr>(
      node_feat_input, weight, node_feat_per_edge, A_col_row_idx_gather_list,
      nullptr, A_col_row_idx_gather_list, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation, num_A_rows, num_A_cols,
      num_B_cols, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputBWProp(
    float* delta_feat_per_edge, float* weight_transposed,
    float* delta_node_input, IdxPtr A_eid_gather_list,
    IdxPtr C_col_row_idx_scatter_list, Idx num_A_rows, Idx num_A_cols,
    Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, false, true, false, false, false, true, false,
                      Idx, IdxPtr>(
      delta_feat_per_edge, weight_transposed, delta_node_input,
      A_eid_gather_list, nullptr, C_col_row_idx_scatter_list, nullptr, nullptr,
      idx_relation, num_A_rows, num_A_cols, num_B_cols, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightBWProp(
    float* node_feat_input, float* delta_feat_per_edge, float* delta_weight,
    IdxPtr A_col_row_idx_gather_list, IdxPtr B_eid_gather_list, Idx num_A_rows,
    Idx num_A_cols, Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, true, true, false, true, false, false, false,
                      Idx, IdxPtr>(
      node_feat_input, delta_feat_per_edge, delta_weight,
      A_col_row_idx_gather_list, B_eid_gather_list, nullptr, nullptr, nullptr,
      idx_relation, num_A_rows, num_A_cols, num_B_cols, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaNodeFeatInputCompactBWProp(
    float* delta_feat_compact, float* weight_transpose,
    float* delta_node_feat_input, IdxPtr A_col_row_idx_gather_list,
    IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_A_rows, Idx num_A_cols,
    Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, false, true, true, false, false, false, false,
                      Idx, IdxPtr>(
      delta_feat_compact, weight_transpose, delta_node_feat_input,
      A_col_row_idx_gather_list, nullptr, nullptr,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      idx_relation, num_A_rows, num_A_cols, num_B_cols, num_heads);
}

template <int BLOCK_SIZE, typename Idx, typename IdxPtr>
__global__ void RGNNDeltaWeightCompactBWProp(
    float* delta_weight, float* feat_input, float* delta_feat_compact,
    IdxPtr A_col_row_idx_gather_list, IdxPtr unique_srcs_and_dests_rel_ptr,
    IdxPtr unique_srcs_and_dests_node_indices, Idx num_A_rows, Idx num_A_cols,
    Idx num_B_cols, Idx num_heads, int* num_blocks_per_relation,
    int* accum_num_blocks_per_relation, Idx num_relations) {
  Idx idx_block_assignment = blockIdx.x / ceil_div<Idx>(num_B_cols, BLOCK_SIZE);
  Idx idx_relation = binary_search<int, int*>(
      num_relations + 1, accum_num_blocks_per_relation, idx_block_assignment);
  _basic_MatMulKernel<BlockSize, true, true, false, true, true, false, false,
                      Idx, IdxPtr>(
      feat_input, delta_feat_compact, delta_weight, A_col_row_idx_gather_list,
      A_col_row_idx_gather_list, nullptr, unique_srcs_and_dests_rel_ptr,
      unique_srcs_and_dests_node_indices, idx_relation, num_A_rows, num_A_cols,
      num_B_cols, num_heads);
}
