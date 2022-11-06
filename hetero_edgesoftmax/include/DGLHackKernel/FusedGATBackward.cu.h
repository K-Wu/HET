#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
struct BackwardGatFusedData {
  // feat_size size along feature dimension
  Idx feat_src_xlen{0};
  Idx feat_src_hidden{0};
  Idx e_xlen{0};
  Idx ret_xlen{0};
  // num nodes
  Idx n{0};
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
__device__ DType gradLeaky(DType val, DType slope) {
  return val > 0 ? 1 : slope;
}

// TODO: test correctness of the fused kernel
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void fusedGatBackwardGradElErFeatSrcFused(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, int64_t num_rows) {
  Idx src_vid = blockIdx.y;
  Idx stride_vid = gridDim.y;
  Idx e_xlen = gdata.e_xlen;
  Idx stride_head = blockDim.x * gridDim.x;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  while (src_vid < num_rows) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (head_idx < e_xlen) {
      Idx feat_idx = threadIdx.y;
      while (feat_idx < hidden_xlen) {
        DType s = 0.;
        DType sfeatsrc = 0.;
        Idx feat_src_offset;
        Idx src_node_feat_offset;
        if constexpr (CompactAsOfNodeFlag) {
          feat_src_offset =
              src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          src_node_feat_offset = src_vid * e_xlen + head_idx;
        }

        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx dst_node_feat_offset;
          if constexpr (!CompactAsOfNodeFlag) {
            feat_src_offset =
                eid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
            dst_node_feat_offset = eid * e_xlen + head_idx;
            src_node_feat_offset = eid * e_xlen + head_idx;
          } else {
            dst_node_feat_offset = dst_vid * e_xlen + head_idx;
          }

          Idx edge_offset = eid * e_xlen + head_idx;

          Idx dst_out_offset =
              dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          DType grad_exp =
              gdata.grad_out[dst_out_offset] *
              (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
              gdata.sum[dst_node_feat_offset];
          DType tmp_sum =
              gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
          DType tmp2 = grad_exp * gdata.exp[edge_offset] *
                       gradLeaky(tmp_sum, gdata.leaky_relu_slope);

          atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
          if constexpr (!CompactAsOfNodeFlag) {
            atomicAdd(gdata.grad_el + src_node_feat_offset, tmp2);
            atomicAdd(gdata.grad_feat_src + feat_src_offset,
                      gdata.exp[eid * e_xlen + head_idx] /
                          gdata.sum[dst_vid * e_xlen + head_idx] *
                          gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {
            sfeatsrc += gdata.exp[eid * e_xlen + head_idx] /
                        gdata.sum[dst_vid * e_xlen + head_idx] *
                        gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                       head_idx * hidden_xlen + feat_idx];
            s += tmp2;
          }
        }
        if constexpr (CompactAsOfNodeFlag) {
          gdata.grad_feat_src[feat_src_offset] = sfeatsrc;
          atomicAdd(gdata.grad_el + src_node_feat_offset, s);
        }

        feat_idx += blockDim.y;
      }
      head_idx += stride_head;
    }
    src_vid += stride_vid;
  }
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
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void fusedGatBackwardGradFeatSrc(
    BackwardGatFusedData<Idx, DType> gdata, const Idx* row_offsets,
    const Idx* column_indices, int64_t num_rows) {
  if constexpr (!CompactAsOfNodeFlag) {
    assert(0 && "not implemented yet");
  }
  Idx src_vid = blockIdx.y;
  Idx stride_vid = gridDim.y;
  Idx e_xlen = gdata.e_xlen;
  Idx stride_head = blockDim.x * gridDim.x;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  while (src_vid < num_rows) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (head_idx < e_xlen) {
      Idx feat_idx = threadIdx.y;
      while (feat_idx < hidden_xlen) {
        DType s = 0.;
        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_id = column_indices[e];
          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?
          s += gdata.exp[eid * e_xlen + head_idx] /
               gdata.sum[dst_id * e_xlen + head_idx] *
               gdata.grad_out[dst_id * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];
        }
        gdata.grad_feat_src[src_vid * gdata.feat_src_xlen +
                            head_idx * hidden_xlen + feat_idx] = s;
        feat_idx += blockDim.y;
      }
      head_idx += stride_head;
    }
    src_vid += stride_vid;
  }
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
template <typename Idx, typename DType, bool CompactAsOfNodeFlag>
__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata,
                                         const Idx* row_offsets,
                                         const Idx* column_indices,
                                         int64_t num_rows) {
  if constexpr (!CompactAsOfNodeFlag) {
    assert(0 && "not implemented yet");
  }
  Idx src_vid = blockIdx.y;
  Idx stride_vid = gridDim.y;
  Idx e_xlen = gdata.e_xlen;
  Idx stride_head = blockDim.x * gridDim.x;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  while (src_vid < num_rows) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (head_idx < e_xlen) {
      Idx feat_idx = threadIdx.y;
      while (feat_idx < hidden_xlen) {
        DType s = 0.;
        Idx feat_src_offset =
            src_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
        Idx src_node_feat_offset = src_vid * e_xlen + head_idx;
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * e_xlen + head_idx;
          Idx dst_vid = column_indices[e];
          Idx dst_node_feat_offset = dst_vid * e_xlen + head_idx;
          Idx dst_out_offset =
              dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen + feat_idx;
          DType grad_exp =
              gdata.grad_out[dst_out_offset] *
              (gdata.feat_src[feat_src_offset] - gdata.ret[dst_out_offset]) /
              gdata.sum[dst_node_feat_offset];
          DType tmp_sum =
              gdata.el[src_node_feat_offset] + gdata.er[dst_node_feat_offset];
          DType tmp2 = grad_exp * gdata.exp[edge_offset] *
                       gradLeaky(tmp_sum, gdata.leaky_relu_slope);
          s += tmp2;
          atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
        }
        atomicAdd(gdata.grad_el + src_node_feat_offset, s);
        feat_idx += blockDim.y;
      }
      head_idx += stride_head;
    }
    src_vid += stride_vid;
  }
}
