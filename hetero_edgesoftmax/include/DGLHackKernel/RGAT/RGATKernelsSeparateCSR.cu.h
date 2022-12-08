#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"

// same vertex-centric schedule as HET_gatSumProdZipDivKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
// The only difference is that etype is now fetched through binary search into
// rel_ptrs, rather than etype array subscription
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void
HET_gatSumProdZipDivKernel_relational_separate_csr_vertex_parallel(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_offsets,
    const Idx* col_indices, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _gatSumProdZipDivKernel<Idx, DType, CompactAsOfNodeFlag, true, true>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}

// same vertex-centric schedule as HET_gatExpLeakyReluSumKernel in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
// The only difference is that etype is now fetched through binary search into
// rel_ptrs, rather than etype array subscription
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void
HET_gatExpLeakyReluSumKernel_relational_separate_csr_vertex_parallel(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_offsets,
    const Idx* col_indices, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _gatExpLeakyReluSumKernel<Idx, DType, CompactAsOfNodeFlag, true, true, false>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}
