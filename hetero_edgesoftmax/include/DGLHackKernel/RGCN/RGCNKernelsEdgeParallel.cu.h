#pragma once

#include <cuda_runtime.h>

template <typename Idx, typename DType>
struct RGCNData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  // Idx feat_src_hidden{0};
  // Idx num_heads{0};
  // Idx ret_xlen{0};
  // num nodes
  // Idx n{0};
  Idx* __restrict__ eids{nullptr};
  // DType leaky_relu_slope;
  // Inputs
  DType* __restrict__ feat_src{nullptr}, *__restrict__ enorm{nullptr};
  // Output
  DType* __restrict__ ret{nullptr};
};

// adapted from _gatSumProdZipDivKernel_edge_parallel in
// [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool FullCartesianFlag>
__device__ __forceinline__ void _rgcnNodeMeanAggregation_edge_parallel(
    RGCNData<Idx, DType> gdata, const Idx* etypes, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  // Idx num_heads = gdata.num_heads;
  // Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx eidx = blockIdx.y; eidx < num_edges; eidx += gridDim.y) {
    Idx dst_vid = col_indices[eidx];

    Idx src_vid = row_indices[eidx];
    // for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y)
    // { Idx start_off = *(row_offsets + dst_vid); Idx end_off = *(row_offsets +
    // dst_vid + 1);
    // for (Idx head_idx = threadIdx.y; head_idx < num_heads;
    //     head_idx += blockDim.y) {
    // it is still using type 2 schedule with num_head == 1
    for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
         feat_idx < gdata.feat_src_xlen; feat_idx += blockDim.x * gridDim.x) {
      // DType s = 0.;
      // for (Idx eidx = start_off; eidx < end_off; eidx++) {
      Idx feat_src_entry_id = -1;
      Idx edata_idx = gdata.eids[eidx];
      if constexpr (RelationalFlag) {
        // Idx sum_idx = -1;
        Idx etype = -1;
        if constexpr (ETypeRelPtrFlag) {
          etype = binary_search(num_relations, etypes, eidx);
        } else {
          etype = etypes[eidx];
        }
        if constexpr (CompactAsOfNodeFlag) {
          feat_src_entry_id = src_vid;
          // sum_idx = find_relational_compact_as_of_node_index(
          //     etype, dst_vid, unique_srcs_and_dests_node_indices,
          //     unique_srcs_and_dests_rel_ptr);
          if constexpr (FullCartesianFlag) {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing along
            // the node dimension may not be worth it.
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                (FullCartesianFlag) && CompactAsOfNodeFlag,
                "should be non-reachable not implemented");
          }
        } else {
          feat_src_entry_id = edata_idx;
          // sum_idx = find_relational_compact_as_of_node_index(
          //     etype, dst_vid, unique_srcs_and_dests_node_indices,
          //     unique_srcs_and_dests_rel_ptr);
        }
        // s += (gdata.enorm[edata_idx] * gdata.feat_src[feat_src_entry_id *
        // gdata.feat_src_xlen +
        //                      /*head_idx * hidden_xlen +*/ feat_idx]);
        atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen +
                             /* head_idx * hidden_xlen +*/ feat_idx],
                  (gdata.enorm[edata_idx] *
                   gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                  /*head_idx * hidden_xlen +*/ feat_idx]));
      } else {  // !RelationalFlag
        feat_src_entry_id = edata_idx;
        // s += gdata.enorm[edata_idx] *
        //      gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen/* +
        //                     head_idx * hidden_xlen*/ + feat_idx];

        atomicAdd(&gdata.ret[dst_vid * gdata.feat_src_xlen +
                             /*head_idx * hidden_xlen + */ feat_idx],
                  gdata.enorm[edata_idx] *
                      gdata.feat_src[feat_src_entry_id * gdata.feat_src_xlen +
                                     /*head_idx * hidden_xlen + */ feat_idx]);
      }

      //}
      // gdata.ret[dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen +
      //           feat_idx] = s;
    }
    //}
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void HET_rgcnNodeMeanAggregation_edge_parallel(
    RGCNData<Idx, DType> gdata, const Idx* rel_ptrs, const Idx* row_indices,
    const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  _rgcnNodeMeanAggregation_edge_parallel<Idx, DType, CompactAsOfNodeFlag, true,
                                         false>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices,
      num_relations);
}