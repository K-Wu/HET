#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer1BackwardCOOKernelEdgePerWarp(
    Idx *src_ids, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes, int edge_idx) {
  Idx tx = threadIdx.x % 32;
  Idx dst_id = __ldg(dst_ids + edge_idx);
  Idx eid = __ldg(eids + edge_idx);
  Idx type_id = __ldg(types + edge_idx);
  Idx src_id = __ldg(src_ids + edge_idx);
  for (; tx < feat_len_x * feat_len_y; tx += 32) {
    Idx ty = tx / feat_len_x;
    Idx th = tx % feat_len_x;
    DType h = __ldg(hidden + src_id * feat_len_y + ty);
    DType agg = 0.;

    DType g = __ldg(grad_out + dst_id * feat_len_x + th);
    DType w = __ldg(weight + type_id * feat_len_y * feat_len_x + tx);
    DType n = __ldg(norm + eid);
    agg += g * w * n;
    atomicAdd(grad_weight + type_id * feat_len_y * feat_len_x + tx, g * h * n);

    atomicAdd(grad_hidden + src_id * feat_len_y + ty, agg);
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer1BackwardCOOKernelImpl(
    Idx *src_ids, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_edges, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes) {
  Idx warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warp_idx < num_edges) {
    RgcnLayer1BackwardCOOKernelEdgePerWarp(
        src_ids, dst_ids, eids, types, hidden, weight, norm, grad_out,
        grad_hidden, grad_weight, num_edges, feat_len_y, feat_len_x, ntypes,
        /* node_idx = */ warp_idx);
  }
}