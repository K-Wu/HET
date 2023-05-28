#pragma once

#include <cuda_runtime.h>

#include "kernel_enums.h"

template <typename Idx, typename DType>
struct RGCNData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx* __restrict__ eids{nullptr};
  // Inputs
  DType* __restrict__ feat_src{nullptr}, * __restrict__ enorm{nullptr};
  // Output
  DType* __restrict__ ret{nullptr};
};

// adapted from _gatSumProdZipDivKernel_edge_parallel in
// [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _rgcnNodeMeanAggregation_edge_parallel(
    RGCNData<Idx, DType> gdata, const Idx* etypes, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx dst_vid = col_indices[eidx];

    Idx src_vid = row_indices[eidx];
    // it is still using type 2 schedule with num_head == 1
    for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
         feat_idx < gdata.feat_src_xlen; feat_idx += blockDim.x * gridDim.x) {
      Idx feat_src_entry_id = -1;
      Idx edata_idx = gdata.eids[eidx];
      if constexpr (RelationalFlag) {
        Idx etype = -1;
        if constexpr (ETypeRelPtrFlag) {
          etype = binary_search(num_relations, etypes, eidx);
        } else {
          etype = etypes[eidx];
        }
        if constexpr (IsCompact(kind)) {
          feat_src_entry_id = src_vid;
          if constexpr (FullCartesianFlag) {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing along
            // the node dimension may not be worth it.
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                (FullCartesianFlag) && IsCompact(kind),
                "should be non-reachable not implemented");
          }
        } else {
          feat_src_entry_id = edata_idx;
        }
        atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen + feat_idx],
                  (gdata.enorm[edata_idx] *
                   gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                  feat_idx]));
      } else {  // !RelationalFlag
        feat_src_entry_id = edata_idx;

        atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen + feat_idx],
                  gdata.enorm[edata_idx] *
                      gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                     feat_idx]);
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void HET_rgcnNodeMeanAggregation_edge_parallel(
    RGCNData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _rgcnNodeMeanAggregation_edge_parallel<Idx, DType, kind, true, false>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}