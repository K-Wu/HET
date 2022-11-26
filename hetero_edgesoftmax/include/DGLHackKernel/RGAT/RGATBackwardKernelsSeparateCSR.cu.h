#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"

// vertex centric schedule similar to fusedGatBackwardGradElErFeatSrcFused in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void
fusedGatBackwardGradElErFeatSrcFused_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_offsets, const Idx* col_indices, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactAsOfNodeFlag, true,
                                        true>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}

// vertex centric schedule similar to fusedGatBackwardGradFeatSrc in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void
fusedGatBackwardGradFeatSrc_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_offsets, const Idx* col_indices, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeFlag, true, true>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}

// vertex centric schedule similar to fusedGatBackwardGradElEr in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void
fusedGatBackwardGradElEr_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_offsets, const Idx* col_indices, const Idx* column_indices,
    int64_t num_rows, const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeFlag, true, true>(
      gdata, row_offsets, col_indices, column_indices, rel_ptrs, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}