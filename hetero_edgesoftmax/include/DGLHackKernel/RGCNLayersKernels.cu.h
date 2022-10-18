#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
// NB: A wrapper version for python api export is implemented at
// hetero_edgesoftmax/src/export/torch/op.cu.cc. Please update accordingly
// whenever there is update.
// TODO: the layer 0 and 1 may ends with bias and activation
// the referential implementation from seastar
// TODO: add load balance kernels
template <typename Idx, typename DType>
__global__ void RgcnLayer0KernelImpl(Idx* ranges, Idx* src_ids, Idx* eids,
                                     Idx* types, DType* weight, DType* norm,
                                     DType* ret, Idx num_nodes, Idx feat_len,
                                     Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_len; tx += blockDim.x) {
      DType agg_val = 0.;
      for (; beg < end; beg++) {
        Idx src_id = __ldg(src_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType w = __ldg(weight + type_id * ntypes * feat_len +
                        src_id * feat_len + tx);
        DType n = __ldg(norm + eid);
        agg_val += w * n;
        // printf("w:%f norm:%f agg_val:%f\n", w, n, agg_val);
      }
      ret[blockIdx.x * feat_len + tx] = agg_val;
    }
  }
}

// bgs:
template <typename Idx, typename DType>
__global__ void RgcnLayer1KernelImpl(const Idx* ranges, const Idx* src_ids,
                                     const Idx* eids, const Idx* types,
                                     const DType* hidden, const DType* weight,
                                     const DType* norm, DType* ret,
                                     Idx num_nodes, Idx feat_len_y,
                                     Idx feat_len_x, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    Idx ty = threadIdx.x / feat_len_x;
    Idx th = threadIdx.x % feat_len_x;
    DType agg_val = 0.;
    DType w = 0.;
    Idx cur_type_id = -1;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
      }
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      DType n = __ldg(norm + eid);
      agg_val += h * w * n;
    }
    atomicAdd(ret + blockIdx.x * feat_len_x + th, agg_val);
  }
}
