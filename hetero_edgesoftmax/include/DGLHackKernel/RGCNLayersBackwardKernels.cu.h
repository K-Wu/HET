#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

// TODO: the layer 0 and 1 may ends with bias and activation
// the referential implementation from seastar

template <typename Idx, typename DType>
__global__ void RgcnLayer0BackwardKernelImpl(Idx *ranges, Idx *dst_ids,
                                             Idx *eids, Idx *types,
                                             DType *grad_out, DType *norm,
                                             DType *grad_weight, Idx num_nodes,
                                             Idx feat_len, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_len; tx += blockDim.x) {
      for (; beg < end; beg++) {
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType w = __ldg(grad_out + dst_id * feat_len + tx);
        DType n = __ldg(norm + eid);
        grad_weight[type_id * ntypes * feat_len + blockIdx.x * feat_len + tx] =
            w * n;
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer1BackwardKernelImpl(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
      Idx ty = tx / feat_len_x;
      Idx th = tx % feat_len_x;
      DType h = __ldg(hidden + blockIdx.x * feat_len_y + ty);
      DType agg = 0.;
      for (; beg < end; beg++) {
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType g = __ldg(grad_out + dst_id * feat_len_x + th);
        DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
        DType n = __ldg(norm + eid);
        agg += g * w * n;
        atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx,
                  g * h * n);
      }
      atomicAdd(grad_hidden + blockIdx.x * feat_len_y + ty, agg);
    }
  }
}
