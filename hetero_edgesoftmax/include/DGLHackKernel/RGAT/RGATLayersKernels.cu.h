#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/GAT/FusedGAT.cu.h"

template <typename Idx, typename DType>
__global__ void
gatSumProdZipDivKernel_relational_separate_coo_compact_as_of_node(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_rows) {}

template <typename Idx, typename DType>
__global__ void
gatExpLeakyReluSumKernel_relational_separate_coo_compact_as_of_node(
    GatFusedData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_rows) {}
