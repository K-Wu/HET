#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
struct GatFusedData {
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
  // Intermediates
  DType *sum{nullptr}, *exp{nullptr};
  // Output
  DType* ret{nullptr};
};

template <typename DType>
__device__ DType gatLeakyReluExp(DType val, DType slope) {
  return val > 0 ? exp(val) : exp(slope * val);
}

template <typename Idx, typename DType>
__global__ void gatSumProdZipDivKernel(GatFusedData<Idx, DType> gdata,
                                       const Idx* row_offsets,
                                       const Idx* column_indices,
                                       int64_t num_rows) {
  Idx dst_vid = blockIdx.y;
  Idx stride_vid = gridDim.y;
  Idx stride_head = blockDim.x * gridDim.x;
  Idx e_xlen = gdata.e_xlen;
  Idx hidden_xlen = gdata.feat_src_xlen / e_xlen;
  while (dst_vid < num_rows) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (head_idx < e_xlen) {
      Idx feat_idx = threadIdx.y;
      while (feat_idx < hidden_xlen) {
        DType s = 0.;
        for (Idx eid = start_off; eid < end_off; eid++) {
          Idx src_vid = column_indices[eid];
          // s +=  gdata.exp[gdata.eids[eid] * e_xlen + head_idx] /
          // gdata.sum[dst_vid*e_xlen + head_idx]
          s += gdata.exp[eid * e_xlen + head_idx] /
               gdata.sum[dst_vid * e_xlen + head_idx] *
               gdata.feat_src[src_vid * gdata.feat_src_xlen +
                              head_idx * hidden_xlen + feat_idx];
        }
        gdata.ret[dst_vid * gdata.feat_src_xlen + head_idx * hidden_xlen +
                  feat_idx] = s;
        feat_idx += blockDim.y;
      }
      head_idx += stride_head;
    }
    dst_vid += stride_vid;
  }
}

template <typename Idx, typename DType>
__global__ void gatExpLeakyReluSumKernel(GatFusedData<Idx, DType> gdata,
                                         const Idx* row_offsets,
                                         const Idx* column_indices,
                                         int64_t num_rows) {
  // extern __shared__ DType er[];
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx stride_x = blockDim.x * gridDim.x;
  Idx stride_y = blockDim.y * gridDim.y;
  Idx dst_vid = ty;
  Idx e_xlen = gdata.e_xlen;
  while (dst_vid < num_rows) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    Idx feat_idx = tx;
    while (feat_idx < e_xlen) {
      // 1. Load dstnation vertex into shared memory
      Idx feat_off_dst = dst_vid * e_xlen + feat_idx;
      // er[threadIdx.x] = gdata.er[feat_off_dst];
      //__syncthreads();
      // 2. Do the computation
      DType sum = 0.;
      for (Idx eid = start_off; eid < end_off; ++eid) {
        Idx src_id = *(column_indices + eid);
        Idx feat_off_src = src_id * e_xlen + feat_idx;
        // DType tmp = gatLeakyReluExp(gdata.el[feat_off_src] + er[threadIdx.x],
        // gdata.leaky_relu_slope);
        DType tmp =
            gatLeakyReluExp(gdata.el[feat_off_src] + gdata.er[feat_off_dst],
                            gdata.leaky_relu_slope);
        // gdata.exp[Idx(gdata.eids[eid] * e_xlen) + feat_idx] = tmp;
        gdata.exp[Idx(eid * e_xlen) + feat_idx] = tmp;

        sum += tmp;
      }
      gdata.sum[Idx(dst_vid * e_xlen) + feat_idx] = sum;
      feat_idx += stride_x;
    }
    dst_vid += stride_y;
  }
}