#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
struct BackwardGatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  // Idx feat_src_hidden{0};
  Idx num_heads{0};
  // Idx ret_xlen{0};
  // num nodes
  // Idx n{0};
  Idx* eids;
  DType leaky_relu_slope;
  // Inputs
  DType *feat_src{nullptr}, *el{nullptr}, *er{nullptr};
  DType *sum{nullptr}, *exp{nullptr}, *ret{nullptr};
  // Output
  DType *grad_out{nullptr}, *grad_feat_src{nullptr}, *grad_el{nullptr},
      *grad_er{nullptr};
};

template <typename DType>
__device__ __forceinline__ DType gradLeaky(DType val, DType slope) {
  return val > 0 ? 1 : slope;
}

// TODO: check if outcsr + incsr is correctly applied to each kernel
// TODO: test correctness of the fused kernel
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradElErFeatSrcFused(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
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
        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
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
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              er_idx = dst_vid_relational * num_heads + head_idx;
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              el_idx = src_vid_relational * num_heads + head_idx;

              feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx;
              // printf(
              //     "src_vid %ld dst_vid %ld etype %ld src_vid_relational %ld "
              //     "dst_vid_relational %ld \n",
              //     src_vid, dst_vid, etype, src_vid_relational,
              //     dst_vid_relational);
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
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_el + el_idx, tmp2);
            atomicAdd(gdata.grad_feat_src + feat_src_offset,
                      gdata.exp[eid * num_heads + head_idx] /
                          gdata.sum[dst_vid * num_heads + head_idx] *
                          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {
            sfeatsrc += gdata.exp[eid * num_heads + head_idx] /
                        gdata.sum[dst_vid * num_heads + head_idx] *
                        gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                       head_idx * hidden_xlen + feat_idx];
            s += tmp2;
          }  // if constexpr (!CompactAsOfNodeFlag)
        }    // for Idx e
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
          atomicAdd(gdata.grad_el + el_idx, s);
        }
      }  // while feat_idx
    }    // while head_idx
  }      // while src_vid
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradElErFeatSrcFused(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices) {
  _fusedGatBackwardGradElErFeatSrcFused<Idx, DType, CompactAsOfNodeFlag,
                                        RelationalFlag, false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
/*** Implement the logic of computing grad_feat_src.
    feat_src is of dimension: N * num_heads * num_hidden
    exp is of dimension: M * num_heads
    sum is of dimension: N * num_heads
    * means element-wise mutliplication
    In forward computation: out = sum([feat_src[e.src] * exp[e.eid]/sum[curnode]
for e in curnode.inedges]), In backward computation: grad_feat_src[curnode] =
sum([grad_out[e.dst] * exp[e.eid]/sum[e.dst] for e in curnode.outedges])
***/
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradFeatSrc(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
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
        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
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
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }
          }
          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?
          // Idx sum_vid = dst_vid;
          // if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
          //   sum_vid = dst_vid_relational;
          // }
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_feat_src + feat_src_offset,
                      gdata.exp[eid * num_heads + head_idx] /
                          gdata.sum[dst_vid * num_heads + head_idx] *
                          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            s += gdata.exp[eid * num_heads + head_idx] /
                 gdata.sum[dst_vid * num_heads + head_idx] *
                 gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = s;
        }
      }
    }
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradFeatSrc(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices) {
  _fusedGatBackwardGradFeatSrc<Idx, DType, CompactAsOfNodeFlag, RelationalFlag,
                               false>(
      gdata, row_offsets, column_indices, etypes, num_rows,
      unique_srcs_and_dests_rel_ptr, unique_srcs_and_dests_node_indices, -1);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
/***
Implement the logic of computing grad_el.
Dimension of grad_out: N * num_heads * num_hidden
             grad_el:  N * num_heads
             grad_er:  N * num_heads
             el:       N * num_heads
             er:       N * num_heads
             exp:      M * num_heads
             sum:      N * num_heads
             feat_src: N * num_heads * num_hidden

In forward computation: gdata.exp = [exp(leaky_relu(e.el[src] + e.el[dst])) for
e in curnode.inedges] gdata.sum[curnode] = sum([exp[e.eid] for e in
curnode.inedges]) out[curnode] = sum([gdata.exp[e.eid] / gdata.sum[curnode] *
gdata.feat_src[e.src] for e in curnode.inedges]) In backward computation:
                        grad_er = sum([grad_exp[e.eid] *
exp(leaky_relu(gdata.el[src]+ gdata.er[dst])) * grad_leaky_relu(gdata.el[src] +
gdata.er[dst]) for e in curnode.inedges]) grad_el = sum([grad_exp[e.eid] *
leaky_relu(gdata.el[src] + gdata.er[dst]) * grad_leaky_relu(gdata.el[src] +
gdata.er[dst]) for e in curnode.outedges]) grad_exp = [grad_out[e.dst] *
(feat_src[e.src] - out[e.dst])/sum[e.dst] for e in outedges]
***/
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradElEr(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  if constexpr (!CompactAsOfNodeFlag) {
    CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(CompactAsOfNodeFlag,
                                       "not implemented yet");
  }
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
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
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
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
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
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
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_el + el_idx, tmp2);
          }
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          atomicAdd(gdata.grad_el + el_idx, s);
        }
      }
    }
  }
}

template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradElEr(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, const Idx* etypes, int64_t num_rows,
    const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices) {
  _fusedGatBackwardGradElEr<Idx, DType, CompactAsOfNodeFlag, RelationalFlag,
                            false>(gdata, row_offsets, column_indices, etypes,
                                   num_rows, unique_srcs_and_dests_rel_ptr,
                                   unique_srcs_and_dests_node_indices, -1);
}

template <typename Idx, typename DType>
constexpr auto relational_fusedGatBackwardGradElEr_per_edge =
    HET_fusedGatBackwardGradElEr<Idx, DType, false, true>;
template <typename Idx, typename DType>
constexpr auto relational_fusedGatBackwardGradFeatSrc_per_edge =
    HET_fusedGatBackwardGradFeatSrc<Idx, DType, false, true>;
template <typename Idx, typename DType>
constexpr auto relational_fusedGatBackwardGradElErFeatSrcFused_per_edge =
    HET_fusedGatBackwardGradElErFeatSrcFused<Idx, DType, false, true>;