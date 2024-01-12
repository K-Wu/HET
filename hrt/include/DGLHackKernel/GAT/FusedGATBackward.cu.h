#pragma once

#include <cuda_runtime.h>

template <typename Idx, typename DType>
struct BackwardGatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType leaky_relu_slope;
  // Inputs
  DType *__restrict__ feat_src{nullptr}, *__restrict__ el{nullptr},
      *__restrict__ er{nullptr};
  DType *__restrict__ sum{nullptr}, *__restrict__ exp{nullptr},
      *__restrict__ ret{nullptr};
  // Output
  DType *__restrict__ grad_out{nullptr}, *__restrict__ grad_feat_src{nullptr},
      *__restrict__ grad_el{nullptr}, *__restrict__ grad_er{nullptr};
};

template <typename DType>
__device__ __forceinline__ DType gradLeaky(DType val, DType slope) {
  return val > 0 ? 1 : slope;
}

// TODO: check if outcsr + incsr is correctly applied to each kernel
// TODO: test correctness of the fused kernel
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradElErFeatSrcFused(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.feat_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      // TODO: switch the loop order of feat_idx and e
      // TODO: think about ways to reduce atomic operations
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        DType sfeatsrc = 0.;
        Idx feat_src_offset = -1;
        Idx el_idx = -1;
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          el_idx = src_vid * num_heads + head_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx er_idx = -1;
          Idx dst_vid_relational = -1;
          if constexpr (!IsCompact(kind)) {
            // in this case, feat_src_offset, er_idx and el_idx are related to
            // edge id, regardless of the type of the edge
            feat_src_offset = edata_idx * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
            er_idx = edata_idx * num_heads + head_idx;
            el_idx = edata_idx * num_heads + head_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              er_idx = dst_vid * num_heads + head_idx;
            } else {  // RelationalFlag
              // in this case, er_idx (sum's index) is related to (relation,
              // unique node index) el_idx is related to (relation, unique node
              // index) feat_src_offset is related to (relation, unique node
              // index)
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, e);
              } else {
                etype = etype_data.etypes[e];
              }
              // TODO: etype is not needed if etype_mapper_data
              // !IsBinarySearch(kind)
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, edata_idx, etype_mapper_data);
              er_idx = dst_vid_relational * num_heads + head_idx;
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);
              el_idx = src_vid_relational * num_heads + head_idx;

              feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx;
            }
          }

          Idx edata_offset = edata_idx * num_heads + head_idx;

          Idx dst_out_offset =
              dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          DType grad_exp =
              gdata.grad_out[dst_out_offset] *
              (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
              gdata.sum[dst_vid * num_heads + head_idx];
          DType tmp_sum = gdata.el[el_idx] + gdata.er[er_idx];
          DType tmp2 = grad_exp * gdata.exp[edata_offset] *
                       gradLeaky(tmp_sum, gdata.leaky_relu_slope);

          atomicAdd(gdata.grad_er + er_idx, tmp2);
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            // TODO: no need to use atomicAdd for grad_el
            atomicAdd(gdata.grad_el + el_idx, tmp2);
            // TODO: use csr to remove atomicAdd for grad_feat_src
            atomicAdd(gdata.grad_feat_src + feat_src_offset,
                      gdata.exp[edata_idx * num_heads + head_idx] /
                          gdata.sum[dst_vid * num_heads + head_idx] *
                          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {
            sfeatsrc += gdata.exp[edata_idx * num_heads + head_idx] /
                        gdata.sum[dst_vid * num_heads + head_idx] *
                        gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                       head_idx * hidden_xlen + feat_idx];
            s += tmp2;
          }
        }  // for Idx e
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
          // TODO: no need to use atomicAdd for grad_el
          atomicAdd(gdata.grad_el + el_idx, s);
        }
      }  // while feat_idx
    }    // while head_idx
  }      // while src_vid
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradElErFeatSrcFused(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, false> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  _fusedGatBackwardGradElErFeatSrcFused<Idx, DType, kind, RelationalFlag,
                                        false>(gdata, row_offsets,
                                               column_indices, etype_data,
                                               num_rows, etype_mapper_data);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
/*** Implement the logic of computing grad_feat_src.
    feat_src is of dimension: N * num_heads * num_hidden
    exp is of dimension: M * num_heads
    sum is of dimension: N * num_heads
    * means element-wise mutliplication
    In forward computation: out = sum([feat_src[e.src] *
exp[e.edata_idx]/sum[curnode] for e in curnode.inedges]), In backward
computation: grad_feat_src[curnode] = sum([grad_out[e.dst] *
exp[e.edata_idx]/sum[e.dst] for e in curnode.outedges])
***/
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradFeatSrc(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
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
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx dst_vid_relational = -1;
          if constexpr (!IsCompact(kind)) {
            // in this case, feat_src_offset, er_idx and el_idx are related to
            // edge id, regardless of the type of the edge
            feat_src_offset = edata_idx * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, e);
              } else {  // !ETypeRelPtrFlag
                etype = etype_data.etypes[e];
              }
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);
              feat_src_offset = src_vid_relational * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx;
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, edata_idx, etype_mapper_data);
            }
          }
          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_feat_src + feat_src_offset,
                      gdata.exp[edata_idx * num_heads + head_idx] /
                          gdata.sum[dst_vid * num_heads + head_idx] *
                          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            s += gdata.exp[edata_idx * num_heads + head_idx] /
                 gdata.sum[dst_vid * num_heads + head_idx] *
                 gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          gdata.grad_feat_src[feat_src_offset] = s;
        }
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradFeatSrc(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, false> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  _fusedGatBackwardGradFeatSrc<Idx, DType, kind, RelationalFlag, false>(
      gdata, row_offsets, column_indices, etype_data, num_rows,
      etype_mapper_data);
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
e in curnode.inedges] gdata.sum[curnode] = sum([exp[e.edata_idx] for e in
curnode.inedges]) out[curnode] = sum([gdata.exp[e.edata_idx] /
gdata.sum[curnode] * gdata.feat_src[e.src] for e in curnode.inedges]) In
backward computation: grad_er = sum([grad_exp[e.edata_idx] *
exp(leaky_relu(gdata.el[src]+ gdata.er[dst])) * grad_leaky_relu(gdata.el[src] +
gdata.er[dst]) for e in curnode.inedges]) grad_el = sum([grad_exp[e.edata_idx] *
leaky_relu(gdata.el[src] + gdata.er[dst]) * grad_leaky_relu(gdata.el[src] +
gdata.er[dst]) for e in curnode.outedges]) grad_exp = [grad_out[e.dst] *
(feat_src[e.src] - out[e.dst])/sum[e.dst] for e in outedges]
***/
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__device__ __forceinline__ void _fusedGatBackwardGradElEr(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const ETypeData<Idx, ETypeRelPtrFlag> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
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
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, feat_src_offset is the same regardless of which
          // outgoing edge we deal with
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          el_idx = src_vid * num_heads + head_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_offset = gdata.eids[e] * num_heads + head_idx;
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx er_idx = -1;
          Idx dst_vid_relational = -1;
          if constexpr (!IsCompact(kind)) {
            // in this case, feat_src_offset, er_idx and el_idx are related to
            // edge id, regardless of the type of the edge
            feat_src_offset = edata_idx * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx;
            er_idx = edata_idx * num_heads + head_idx;
            el_idx = edata_idx * num_heads + head_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              er_idx = dst_vid * num_heads + head_idx;
            } else {
              // in this case, er_idx (sum's index) is related to (relation,
              // unique node index) el_idx is related to (relation, unique node
              // index) feat_src_offset is related to (relation, unique node
              // index)
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, e);
              } else {
                etype = etype_data.etypes[e];
              }
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, edata_idx, etype_mapper_data);
              er_idx = dst_vid_relational * num_heads + head_idx;
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);
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
          DType tmp2 = grad_exp * gdata.exp[edata_offset] *
                       gradLeaky(tmp_sum, gdata.leaky_relu_slope);
          s += tmp2;
          atomicAdd(gdata.grad_er + er_idx, tmp2);
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_el + el_idx, tmp2);
          }
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          atomicAdd(gdata.grad_el + el_idx, s);
        }
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag>
__global__ void HET_fusedGatBackwardGradElEr(
    BackwardGatFusedData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, ETypeData<Idx, false> etype_data,
    int64_t num_rows, const ETypeMapperData<Idx, kind> etype_mapper_data) {
  _fusedGatBackwardGradElEr<Idx, DType, kind, RelationalFlag, false>(
      gdata, row_offsets, column_indices, etype_data, num_rows,
      etype_mapper_data);
}
