#pragma once
// #include "DGLHackKernel/DGLHackKernel.h"
#include <cuda_runtime.h>
#include "DGLHackKernel/GAT/FusedGATBackward.cu.h"

// edge-centric schedule cf. HET_fusedGatBackwardGradElErFeatSrcFused in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool DualUniqueNodeList>
__device__ __forceinline__ void
_fusedGatBackwardGradElErFeatSrcFused_edge_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* etypes,
    const Idx* row_indices, const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx e = blockIdx.y; e < num_edges; e += gridDim.y) {
    // for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y)
    // {
    // Idx start_off = row_offsets[src_vid];
    // Idx end_off = row_offsets[src_vid + 1];
    Idx src_vid = row_indices[e];
    Idx eid = gdata.eids[e];
    Idx dst_vid = col_indices[e];

    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        DType sfeatsrc = 0.;
        Idx feat_src_offset = -1;
        Idx el_idx = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          el_idx = src_vid * num_heads + head_idx;
        }
        // for (Idx e = start_off; e < end_off; ++e) {
        Idx er_idx = -1;
        Idx dst_vid_relational = -1;
        if constexpr (!CompactAsOfNodeFlag) {
          // in this case, feat_src_offset, er_idx and el_idx are related to
          // edge id, regardless of the type of the edge
          feat_src_offset =
              eid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          er_idx = eid * num_heads + head_idx;
          el_idx = eid * num_heads + head_idx;
        } else {  // CompactAsOfNodeFlag
          if constexpr (!RelationalFlag) {
            er_idx = dst_vid * num_heads + head_idx;
          } else {  // RelationalFlag
            // in this case, er_idx (sum's index) is related to (relation,
            // unique node index) el_idx is related to (relation, unique node
            // index) feat_src_offset is related to (relation, unique node
            // index)
            // Idx etype = etypes[e];
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, e);
            } else {
              etype = etypes[e];
            }
            if constexpr (DualUniqueNodeList) {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr_col,
                  unique_srcs_and_dests_node_indices_col);
            } else {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }
            er_idx = dst_vid_relational * num_heads + head_idx;
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_vid, unique_srcs_and_dests_rel_ptr,
                unique_srcs_and_dests_node_indices);
            el_idx = src_vid_relational * num_heads + head_idx;
            feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
          }
        }

        Idx edge_offset = eid * num_heads + head_idx;

        Idx dst_out_offset =
            dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        DType grad_exp =
            gdata.grad_out[dst_out_offset] *
            (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
            gdata.sum[dst_vid * num_heads + head_idx];
        DType tmp_sum = gdata.el[el_idx] + gdata.er[er_idx];
        DType tmp2 = grad_exp * gdata.exp[edge_offset] *
                     gradLeaky(tmp_sum, gdata.leaky_relu_slope);

        atomicAdd(gdata.grad_er + er_idx, tmp2);
        // Idx sum_vid = dst_vid;
        // if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
        //   sum_vid = dst_vid_relational;
        // }
        // if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
        atomicAdd(gdata.grad_el + el_idx, tmp2);
        atomicAdd(gdata.grad_feat_src + feat_src_offset,
                  gdata.exp[eid * num_heads + head_idx] /
                      gdata.sum[dst_vid * num_heads + head_idx] *
                      gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                     head_idx * hidden_xlen + feat_idx]);
        //   } else {
        //     sfeatsrc += gdata.exp[eid * num_heads + head_idx] /
        //                 gdata.sum[sum_vid * num_heads + head_idx] *
        //                 gdata.grad_out[dst_vid * gdata.feat_src_xlen +
        //                                head_idx * hidden_xlen + feat_idx];
        //     s += tmp2;
        //   }  // if constexpr (!CompactAsOfNodeFlag)
        //}    // for Idx e
        // if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
        //   gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
        //   atomicAdd(gdata.grad_el + el_idx, s);
        // }
      }  // while feat_idx
    }    // while head_idx
  }      // while src_vid
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool DualUniqueNodeList>
__global__ void
HET_fusedGatBackwardGradElErFeatSrcFused_relational_separate_coo(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  _fusedGatBackwardGradElErFeatSrcFused_edge_parallel<
      Idx, DType, CompactAsOfNodeFlag, true, DualUniqueNodeList>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_rel_ptr_col,
      unique_srcs_and_dests_node_indices,
      unique_srcs_and_dests_node_indices_col, num_relations);
}

// edge-centric schedule cf. HET_fusedGatBackwardGradFeatSrc in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool DualUniqueNodeList>
__device__ __forceinline__ void _fusedGatBackwardGradFeatSrc_edge_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* etypes,
    const Idx* row_indices, const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx e = blockIdx.y; e < num_edges; e += gridDim.y) {
    Idx src_vid = row_indices[e];

    Idx eid = gdata.eids[e];
    Idx dst_vid = col_indices[e];
    // for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y)
    // {
    // Idx start_off = row_offsets[src_vid];
    // Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        Idx feat_src_offset = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        }
        // for (Idx e = start_off; e < end_off; ++e) {
        Idx dst_vid_relational = -1;
        if constexpr (!CompactAsOfNodeFlag) {
          // in this case, feat_src_offset, er_idx and el_idx are related to
          // edge id, regardless of the type of the edge
          feat_src_offset =
              eid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        } else {  // CompactAsOfNodeFlag
          if constexpr (RelationalFlag) {
            // Idx etype = etypes[e];
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, e);
            } else {  // !ETypeRelPtrFlag
              etype = etypes[e];
            }
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_vid, unique_srcs_and_dests_rel_ptr,
                unique_srcs_and_dests_node_indices);
            feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
            if constexpr (DualUniqueNodeList) {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr_col,
                  unique_srcs_and_dests_node_indices_col);
            } else {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }
          }
        }
        // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
        // well as redundant computation?
        // Idx sum_vid = dst_vid;
        // if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
        //   sum_vid = dst_vid_relational;
        // }
        // if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
        atomicAdd(gdata.grad_feat_src + feat_src_offset,
                  gdata.exp[eid * num_heads + head_idx] /
                      gdata.sum[dst_vid * num_heads + head_idx] *
                      gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                     head_idx * hidden_xlen + feat_idx]);
        //   } else {  // CompactAsOfNodeFlag && !RelationalFlag
        //     s += gdata.exp[eid * num_heads + head_idx] /
        //          gdata.sum[sum_vid * num_heads + head_idx] *
        //          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
        //                         head_idx * hidden_xlen + feat_idx];
        //   }
        //}
        // if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
        //   gdata.grad_feat_src[feat_src_offset] = s;
        // }
      }
    }
  }
}

// edge-centric schedule cf. HET_fusedGatBackwardGradElEr in
// [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGATBackward.cu.h]]
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool DualUniqueNodeList>
__device__ __forceinline__ void _fusedGatBackwardGradElEr_edge_parallel(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* etypes,
    const Idx* row_indices, const Idx* column_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  constexpr bool ETypeRelPtrFlag = true;
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx e = blockIdx.y; e < num_edges; e += gridDim.y) {
    Idx src_vid = row_indices[e];

    Idx eid = gdata.eids[e];
    Idx dst_vid = column_indices[e];
    // for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y)
    // {
    //  Idx start_off = row_offsets[src_vid];
    // Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        Idx feat_src_offset = -1;
        Idx el_idx = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          el_idx = src_vid * num_heads + head_idx;
        }
        // for (Idx e = start_off; e < end_off; ++e) {
        Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
        Idx er_idx = -1;
        Idx dst_vid_relational = -1;
        if constexpr (!CompactAsOfNodeFlag) {
          // in this case, feat_src_offset, er_idx and el_idx are related to
          // edge id, regardless of the type of the edge
          feat_src_offset =
              eid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          er_idx = eid * num_heads + head_idx;
          el_idx = eid * num_heads + head_idx;
        } else {  // CompactAsOfNodeFlag
          if constexpr (!RelationalFlag) {
            er_idx = dst_vid * num_heads + head_idx;
          } else {
            // in this case, er_idx (sum's index) is related to (relation,
            // unique node index) el_idx is related to (relation, unique node
            // index) feat_src_offset is related to (relation, unique node
            // index)
            // Idx etype = etypes[e];
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, e);
            } else {
              etype = etypes[e];
            }
            if constexpr (DualUniqueNodeList) {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr_col,
                  unique_srcs_and_dests_node_indices_col);
            } else {
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }

            er_idx = dst_vid_relational * num_heads + head_idx;
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_vid, unique_srcs_and_dests_rel_ptr,
                unique_srcs_and_dests_node_indices);
            el_idx = src_vid_relational * num_heads + head_idx;
            feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
          }
        }
        Idx dst_out_offset =
            dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        DType grad_exp =
            gdata.grad_out[dst_out_offset] *
            (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
            gdata.sum[dst_vid * num_heads + head_idx];
        DType tmp_sum = gdata.el[el_idx] + gdata.er[er_idx];
        DType tmp2 = grad_exp * gdata.exp[edge_offset] *
                     gradLeaky(tmp_sum, gdata.leaky_relu_slope);
        s += tmp2;
        atomicAdd(gdata.grad_er + er_idx, tmp2);
        // if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
        atomicAdd(gdata.grad_el + el_idx, tmp2);
        //}
        //}
        // if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
        //   atomicAdd(gdata.grad_el + el_idx, s);
        // }
      }
    }
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool DualUniqueNodeList>
__global__ void HET_fusedGatBackwardGradFeatSrc_relational_separate_coo(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* col_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  _fusedGatBackwardGradFeatSrc_edge_parallel<Idx, DType, CompactAsOfNodeFlag,
                                             true, DualUniqueNodeList>(
      gdata, rel_ptrs, row_indices, col_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_rel_ptr_col,
      unique_srcs_and_dests_node_indices,
      unique_srcs_and_dests_node_indices_col, num_relations);
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool DualUniqueNodeList>
__global__ void HET_fusedGatBackwardGradElEr_relational_separate_coo(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* rel_ptrs,
    const Idx* row_indices, const Idx* column_indices, int64_t num_edges,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_rel_ptr_col,
    const Idx* unique_srcs_and_dests_node_indices,
    const Idx* unique_srcs_and_dests_node_indices_col, int64_t num_relations) {
  _fusedGatBackwardGradElEr_edge_parallel<Idx, DType, CompactAsOfNodeFlag, true,
                                          DualUniqueNodeList>(
      gdata, rel_ptrs, row_indices, column_indices, num_edges,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_rel_ptr_col,
      unique_srcs_and_dests_node_indices,
      unique_srcs_and_dests_node_indices_col, num_relations);
}