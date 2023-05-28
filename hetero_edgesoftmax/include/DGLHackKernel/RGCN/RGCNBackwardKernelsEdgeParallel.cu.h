#pragma once

#include <cuda_runtime.h>

template <typename Idx, typename DType>
struct BackwardRGCNData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  // Inputs
  DType *__restrict__ feat_src{nullptr};
  DType *__restrict__ enorm{nullptr};
  DType *__restrict__ ret{nullptr};
  // Output
  DType *__restrict__ grad_out{nullptr}, *__restrict__ grad_feat_src{nullptr};
};

// adapted from _fusedGatBackwardGradElErFeatSrcFused_edge_parallel in
// [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATBackwardKernelsSeparateCOO.cu.h]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__device__ __forceinline__ void _rgcnBackwardNodeMeanAggregation_edge_parallel(
    BackwardRGCNData<Idx, DType> gdata, const Idx *etypes,
    const Idx *row_indices, const Idx *col_indices, int64_t num_edges,
    ETypeMapperData<Idx, kind> etype_mapper_data, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  for (Idx e = blockIdx.y; e < num_edges; e += gridDim.y) {
    Idx src_vid = row_indices[e];
    Idx edata_idx = gdata.eids[e];
    Idx dst_vid = col_indices[e];
    // it is still using type 2 schedule with num_head == 1
    for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
         feat_idx < gdata.feat_src_xlen; feat_idx += blockDim.x * gridDim.x) {
      Idx feat_src_offset = -1;
      if constexpr (IsCompact(kind) && !RelationalFlag) {
        // in this case, feat_src_offset is the same regardless of which
        // outgoing edge we deal with
        feat_src_offset = src_vid * gdata.feat_src_xlen + feat_idx;
      }
      if constexpr (!IsCompact(kind)) {
        // in this case, feat_src_offset, er_idx and el_idx are related to
        // edge id, regardless of the type of the edge
        feat_src_offset = edata_idx * gdata.feat_src_xlen + feat_idx;
      } else {  // CompactAsOfNodeFlag
        if constexpr (!RelationalFlag) {
        } else {  // RelationalFlag
          // in this case, er_idx (sum's index) is related to (relation,
          // unique node index) el_idx is related to (relation, unique node
          // index) feat_src_offset is related to (relation, unique node
          // index)
          Idx etype = -1;
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(num_relations, etypes, e);
          } else {
            etype = etypes[e];
          }
          Idx src_vid_relational = find_relational_compact_as_of_node_index(
              etype, src_vid, edata_idx, etype_mapper_data);
          feat_src_offset = src_vid_relational * gdata.feat_src_xlen + feat_idx;
        }
      }

      Idx dst_out_offset = dst_vid * gdata.feat_src_xlen + feat_idx;

      atomicAdd(gdata.grad_feat_src + feat_src_offset,
                gdata.enorm[edata_idx] *
                    gdata.grad_out[dst_vid * gdata.feat_src_xlen + feat_idx]);

    }  // while feat_idx
  }    // while src_vid
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind>
__global__ void HET_rgcnBackwardNodeMeanAggregation_edge_parallel(
    BackwardRGCNData<Idx, DType> gdata, const Idx *rel_ptrs,
    const Idx *row_indices, const Idx *col_indices, int64_t num_edges,
    ETypeMapperData<Idx, kind> etype_mapper_data, int64_t num_relations) {
  _rgcnBackwardNodeMeanAggregation_edge_parallel<Idx, DType, kind, true>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges, etype_mapper_data,
      num_relations);
}
