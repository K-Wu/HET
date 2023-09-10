#pragma once

#include <cuda_runtime.h>

#include "kernel_enums.h"

// adapted from HET_inner_product_fw_kernel_edge_parallel in
// [[hrt/include/DGLHackKernel/RGNN/InnerProductEdgeParallel.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag>
__global__ void HET_gather_compact_to_non_compact_edge_parallel(
    Idx hidden_xlen, DType *out_data, DType *in_data, const Idx *eids,
    const Idx *row_indices, const Idx *column_indices,
    const ETypeData<Idx, ETypeRelPtrFlag> etype_data, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data) {
  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;

  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx edata_idx = eids[eidx];
    Idx src_vid = column_indices[eidx];

    for (Idx head_idx = threadIdx.y; head_idx < 1; head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;

        Idx feat_src_entry_id = -1;
        if constexpr (RelationalFlag) {
          if constexpr (IsCompact(kind)) {
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              if constexpr (EtypeRelPtrIndexSearch) {
                etype = linear_search(etype_data.num_relations,
                                      etype_data.etypes, eidx, resume_from);
                resume_from = etype;
              } else {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, eidx);
              }
            } else {
              etype = etype_data.etypes[eidx];
            }

            feat_src_entry_id = find_relational_compact_as_of_node_index(
                etype, src_vid, edata_idx, etype_mapper_data);

          } else {
            // NB: we need to use edata_idx instead of eidx here
            feat_src_entry_id = edata_idx;
          }
          // TODO: actually full cartesian can be applied both to
          // feat_src_entry_id and sum_idx, in future we may need to add an
          // additional FullCartesianFlag to cover all cases
          if constexpr (FullCartesianFlag) {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing along
            // the node dimension may not be worth it.
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                FullCartesianFlag, "should be non-reachable not implemented");
          }

        } else {  // !RelationalFlag
          // NB: feat_src_entry_id varies between edata_idx and src_vid
          // depending on compactasofnodeflag
          if constexpr (IsCompact(kind)) {
            feat_src_entry_id = src_vid;
          } else {
            feat_src_entry_id = edata_idx;
          }
        }

        out_data[edata_idx * hidden_xlen + feat_idx] =
            in_data[feat_src_entry_id * +head_idx * hidden_xlen + feat_idx];
      }
    }
  }
}