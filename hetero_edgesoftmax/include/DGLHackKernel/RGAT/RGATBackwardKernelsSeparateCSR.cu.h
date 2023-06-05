#pragma once

#include <cuda_runtime.h>

#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"
#include "kernel_enums.h"

// vertex centric schedule similar to HET_fusedGatBackwardGradElErFeatSrcFused
// in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void
HET_fusedGatBackwardGradElErFeatSrcFused_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *rel_ptrs,
    const Idx *row_offsets, const Idx *col_indices, int64_t num_rows,
    const ETypeMapperData<Idx, kind> etype_mapper_data, int64_t num_relations) {
  _fusedGatBackwardGradElErFeatSrcFused<Idx, DType, kind, true, true>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows, etype_mapper_data,
      num_relations);
}

// vertex centric schedule similar to HET_fusedGatBackwardGradFeatSrc in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void
HET_fusedGatBackwardGradFeatSrc_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *rel_ptrs,
    const Idx *row_offsets, const Idx *col_indices, int64_t num_rows,
    const ETypeMapperData<Idx, kind> etype_mapper_data, int64_t num_relations) {
  _fusedGatBackwardGradFeatSrc<Idx, DType, kind, true, true>(
      gdata, row_offsets, col_indices, rel_ptrs, num_rows, etype_mapper_data,
      num_relations);
}

// vertex centric schedule similar to HET_fusedGatBackwardGradElEr in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void
HET_fusedGatBackwardGradElEr_relational_separate_csr_vertex_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *rel_ptrs,
    const Idx *row_offsets, const Idx *col_indices, const Idx *column_indices,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data,
    int64_t num_relations) {
  _fusedGatBackwardGradElEr<Idx, DType, kind, true, true>(
      gdata, row_offsets, col_indices, column_indices, rel_ptrs, num_rows,
      etype_mapper_data, num_relations);
}