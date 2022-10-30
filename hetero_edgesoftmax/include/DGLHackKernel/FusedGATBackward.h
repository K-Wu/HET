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
template <typename Idx, typename DType>
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
          Idx eid = gdata.eids[e];
          sfeatsrc += gdata.exp[eid * e_xlen + head_idx] /
                      gdata.sum[dst_vid * e_xlen + head_idx] *
                      gdata.grad_out[dst_vid * gdata.feat_src_xlen +
                                     head_idx * hidden_xlen + feat_idx];

          atomicAdd(gdata.grad_er + dst_node_feat_offset, tmp2);
        }
        gdata.grad_feat_src[src_vid * gdata.feat_src_xlen +
                            head_idx * hidden_xlen + feat_idx] = sfeatsrc;
        atomicAdd(gdata.grad_el + src_node_feat_offset, s);
        feat_idx += blockDim.y;
      }
      head_idx += stride_head;
    }
    src_vid += stride_vid;
  }
}

template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradFeatSrc(
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

template <typename Idx, typename DType>
__global__ void fusedGatBackwardGradElEr(BackwardGatFusedData<Idx, DType> gdata,
                                         const Idx* row_offsets,
                                         const Idx* column_indices,
                                         int64_t num_rows) {
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
