#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"

template <typename Idx, typename DType>
__global__ void
fusedGatBackwardGradElErFeatSrcFused_relational_separate_coo_compact_as_of_node(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* col_indices, int64_t num_rows) {}

template <typename Idx, typename DType>
__global__ void
fusedGatBackwardGradFeatSrc_relational_separate_coo_compact_as_of_node(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* col_indices, int64_t num_rows) {}

template <typename Idx, typename DType>
__global__ void
fusedGatBackwardGradElEr_relational_separate_coo_compact_as_of_node(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* col_indices, const Idx* column_indices,
    int64_t num_rows) {}