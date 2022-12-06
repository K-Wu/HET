#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

// TODO: the layer 0 and 1 may ends with bias and activation
// the referential implementation from seastar

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer0BackwardKernelNodePerWarp(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *grad_out,
    DType *norm, DType *grad_weight, Idx num_nodes, Idx feat_len, Idx ntypes,
    int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x % 32;
  for (; tx < feat_len; tx += 32) {
    for (; beg < end; beg++) {
      Idx dst_id = __ldg(dst_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      DType w = __ldg(grad_out + dst_id * feat_len + tx);
      DType n = __ldg(norm + eid);
      grad_weight[type_id * ntypes * feat_len + node_idx * feat_len + tx] =
          w * n;
    }
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer0BackwardKernelNodePerBlock(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *grad_out,
    DType *norm, DType *grad_weight, Idx num_nodes, Idx feat_len, Idx ntypes,
    int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x;
  for (; tx < feat_len; tx += blockDim.x) {
    for (; beg < end; beg++) {
      Idx dst_id = __ldg(dst_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      DType w = __ldg(grad_out + dst_id * feat_len + tx);
      DType n = __ldg(norm + eid);
      grad_weight[type_id * ntypes * feat_len + node_idx * feat_len + tx] =
          w * n;
    }
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer0BackwardKernelImpl(Idx *ranges, Idx *dst_ids,
                                             Idx *eids, Idx *types,
                                             DType *grad_out, DType *norm,
                                             DType *grad_weight, Idx num_nodes,
                                             Idx feat_len, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    RgcnLayer0BackwardKernelNodePerBlock(ranges, dst_ids, eids, types, grad_out,
                                         norm, grad_weight, num_nodes, feat_len,
                                         ntypes, /* node_idx = */ blockIdx.x);
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer0BackwardKernelHybridAssignImpl(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *grad_out,
    DType *norm, DType *grad_weight, Idx num_nodes, Idx feat_len, Idx ntypes,
    int num_blocks_on_blocks_per_node) {
  if (blockIdx.x < num_nodes) {
    RgcnLayer0BackwardKernelNodePerBlock(ranges, dst_ids, eids, types, grad_out,
                                         norm, grad_weight, num_nodes, feat_len,
                                         ntypes, /* node_idx = */ blockIdx.x);

  } else {
    int node_idx =
        threadIdx.x / 32 +
        (blockIdx.x - num_blocks_on_blocks_per_node) * blockDim.x / 32;
    if (node_idx < num_nodes) {
      RgcnLayer0BackwardKernelNodePerWarp(
          ranges, dst_ids, eids, types, grad_out, norm, grad_weight, num_nodes,
          feat_len, ntypes, /* node_idx = */ node_idx);
    }
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer1BackwardKernelNodePerWarp(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes, int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x % 32;
  for (; tx < feat_len_x * feat_len_y; tx += 32) {
    // Idx ty = tx / feat_len_x;
    Idx th = tx % feat_len_x;
    for (Idx ty = tx / feat_len_x; ty < feat_len_y;
         ty += blockDim.x / feat_len_y) {
      DType h = __ldg(hidden + node_idx * feat_len_y + ty);
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
      atomicAdd(grad_hidden + node_idx * feat_len_y + ty, agg);
    }
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer1BackwardKernelNodePerBlock(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes, int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x;
  for (; tx < feat_len_x * feat_len_y; tx += blockDim.x) {
    // Idx ty = tx / feat_len_x;
    Idx th = tx % feat_len_x;
    for (Idx ty = tx / feat_len_x; ty < feat_len_y;
         ty += blockDim.x / feat_len_x) {
      DType h = __ldg(hidden + node_idx * feat_len_y + ty);
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
      atomicAdd(grad_hidden + node_idx * feat_len_y + ty, agg);
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
    RgcnLayer1BackwardKernelNodePerBlock(
        ranges, dst_ids, eids, types, hidden, weight, norm, grad_out,
        grad_hidden, grad_weight, num_nodes, feat_len_y, feat_len_x, ntypes,
        /* node_idx = */ blockIdx.x);
  }
}

template <typename Idx, typename DType>
__global__ void RgcnLayer1BackwardKernelHybridAssignImpl(
    Idx *ranges, Idx *dst_ids, Idx *eids, Idx *types, DType *hidden,
    DType *weight, DType *norm, DType *grad_out, DType *grad_hidden,
    DType *grad_weight, Idx num_nodes, Idx feat_len_y, Idx feat_len_x,
    Idx ntypes, int num_blocks_on_blocks_per_node) {
  if (blockIdx.x < num_blocks_on_blocks_per_node) {
    RgcnLayer1BackwardKernelNodePerBlock(
        ranges, dst_ids, eids, types, hidden, weight, norm, grad_out,
        grad_hidden, grad_weight, num_nodes, feat_len_y, feat_len_x, ntypes,
        /* node_idx = */ blockIdx.x);
  } else {
    int node_idx =
        threadIdx.x / 32 +
        (blockIdx.x - num_blocks_on_blocks_per_node) * blockDim.x / 32;
    if (node_idx < num_nodes) {
      RgcnLayer1BackwardKernelNodePerWarp(
          ranges, dst_ids, eids, types, hidden, weight, norm, grad_out,
          grad_hidden, grad_weight, num_nodes, feat_len_y, feat_len_x, ntypes,
          /* node_idx = */ node_idx);
    }
  }
}