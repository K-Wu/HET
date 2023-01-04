#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
// NB: A wrapper version for python api export is implemented at
// [[hetero_edgesoftmax/src/export/torch/op.cu.cc]]. Please update accordingly
// whenever there is update.
// TODO: the layer 0 and 1 may ends with bias and activation
// the referential implementation from seastar
// TODO: add load balance kernels

template <typename Idx, typename DType>
__device__ __forceinline__ void Seastar_RgcnLayer0KernelNodePerBlock(
    Idx* ranges, Idx* src_ids, Idx* eids, Idx* types, DType* weight,
    DType* norm, DType* ret, Idx num_nodes, Idx feat_len, Idx ntypes,
    int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x;
  for (; tx < feat_len; tx += blockDim.x) {
    DType agg_val = 0.;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      DType w =
          __ldg(weight + type_id * ntypes * feat_len + src_id * feat_len + tx);
      DType n = __ldg(norm + eid);
      agg_val += w * n;
      // printf("w:%f norm:%f agg_val:%f\n", w, n, agg_val);
    }
    ret[node_idx * feat_len + tx] = agg_val;
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void Seastar_RgcnLayer0KernelNodePerWarp(
    Idx* ranges, Idx* src_ids, Idx* eids, Idx* types, DType* weight,
    DType* norm, DType* ret, Idx num_nodes, Idx feat_len, Idx ntypes,
    int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x % 32;
  for (; tx < feat_len; tx += /*WARP_SIZE = */ 32) {
    DType agg_val = 0.;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      DType w =
          __ldg(weight + type_id * ntypes * feat_len + src_id * feat_len + tx);
      DType n = __ldg(norm + eid);
      agg_val += w * n;
      // printf("w:%f norm:%f agg_val:%f\n", w, n, agg_val);
    }
    ret[node_idx * feat_len + tx] = agg_val;
  }
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
template <typename Idx, typename DType>
__global__ void HET_Seastar_RgcnLayer0KernelImpl(Idx* ranges, Idx* src_ids,
                                                 Idx* eids, Idx* types,
                                                 DType* weight, DType* norm,
                                                 DType* ret, Idx num_nodes,
                                                 Idx feat_len, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Seastar_RgcnLayer0KernelNodePerBlock<Idx, DType>(
        ranges, src_ids, eids, types, weight, norm, ret, num_nodes, feat_len,
        ntypes, /*node_idx = */ blockIdx.x);
  }
}

// TODO: export hybrid assign kernels in
// [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGCNOps.inc.h]]
template <typename Idx, typename DType>
__global__ void HET_Seastar_RgcnLayer0KernelHybridAssignImpl(
    Idx* ranges, Idx* src_ids, Idx* eids, Idx* types, DType* weight,
    DType* norm, DType* ret, Idx num_nodes, Idx feat_len, Idx ntypes,
    int num_blocks_on_blocks_per_node) {
  if (blockIdx.x < num_blocks_on_blocks_per_node) {
    Seastar_RgcnLayer0KernelNodePerBlock<Idx, DType>(
        ranges, src_ids, eids, types, weight, norm, ret, num_nodes, feat_len,
        ntypes, /*node_idx = */ blockIdx.x);
  } else {
    int node_idx =
        threadIdx.x / 32 +
        (blockIdx.x - num_blocks_on_blocks_per_node) * blockDim.x / 32;
    if (node_idx < num_nodes) {
      Seastar_RgcnLayer0KernelNodePerWarp<Idx, DType>(
          ranges, src_ids, eids, types, weight, norm, ret, num_nodes, feat_len,
          ntypes, node_idx);
    }
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void Seastar_RgcnLayer1KernelNodePerWarp(
    const Idx* ranges, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_nodes, Idx feat_len_y, Idx feat_len_x, Idx ntypes, int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx lane_idx = threadIdx.x % 32;
  for (int feat_idx = lane_idx; feat_idx < feat_len_x * feat_len_y;
       feat_idx += 32) {
    // Idx ty = feat_idx / feat_len_x;
    Idx th = feat_idx % feat_len_x;
    DType agg_val = 0.;
    DType w = 0.;
    Idx cur_type_id = -1;
    for (; beg < end; beg++) {
      Idx src_id = __ldg(src_ids + beg);
      Idx eid = __ldg(eids + beg);
      Idx type_id = __ldg(types + beg);
      if (type_id != cur_type_id) {
        w = __ldg(weight + type_id * feat_len_y * feat_len_x + feat_idx);
      }
      for (Idx ty = threadIdx.x / feat_len_x; ty < feat_len_y;
           ty += blockDim.x / feat_len_x) {
        DType h = __ldg(hidden + src_id * feat_len_y + ty);
        DType n = __ldg(norm + eid);
        agg_val += h * w * n;
      }
    }
    atomicAdd(ret + node_idx * feat_len_x + th, agg_val);
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void Seastar_RgcnLayer1KernelNodePerBlock(
    const Idx* ranges, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_nodes, Idx feat_len_y, Idx feat_len_x, Idx ntypes, int node_idx) {
  Idx beg = __ldg(ranges + node_idx);
  Idx end = __ldg(ranges + node_idx + 1);
  Idx tx = threadIdx.x;
  // Idx ty = threadIdx.x / feat_len_x;
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
    DType n = __ldg(norm + eid);
    for (Idx ty = threadIdx.x / feat_len_x; ty < feat_len_y;
         ty += blockDim.x / feat_len_x) {
      DType h = __ldg(hidden + src_id * feat_len_y + ty);
      agg_val += h * w * n;
    }
  }
  ret[node_idx * feat_len_x + th] = agg_val;
  // atomicAdd(ret + node_idx * feat_len_x + th, agg_val);
}

// from seastar dgl-hack src/kernel/cuda/binary_reduce_impl.cu
// bgs:
template <typename Idx, typename DType>
__global__ void HET_Seastar_RgcnLayer1KernelImpl(
    const Idx* ranges, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_nodes, Idx feat_len_y, Idx feat_len_x, Idx ntypes) {
  if (blockIdx.x < num_nodes) {
    Seastar_RgcnLayer1KernelNodePerBlock(
        ranges, src_ids, eids, types, hidden, weight, norm, ret, num_nodes,
        feat_len_y, feat_len_x, ntypes, /*node_idx = */ blockIdx.x);
  }
}

template <typename Idx, typename DType>
__global__ void HET_Seastar_RgcnLayer1KernelHybridAssignImpl(
    const Idx* ranges, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_nodes, Idx feat_len_y, Idx feat_len_x, Idx ntypes,
    int num_blocks_on_blocks_per_node) {
  // TODO
  if (blockIdx.x < num_blocks_on_blocks_per_node) {
    Seastar_RgcnLayer1KernelNodePerBlock(
        ranges, src_ids, eids, types, hidden, weight, norm, ret, num_nodes,
        feat_len_y, feat_len_x, ntypes, /*node_idx = */ blockIdx.x);
  } else {
    int node_idx =
        threadIdx.x / 32 +
        (blockIdx.x - num_blocks_on_blocks_per_node) * blockDim.x / 32;
    if (node_idx < num_nodes) {
      Seastar_RgcnLayer1KernelNodePerWarp(
          ranges, src_ids, eids, types, hidden, weight, norm, ret, num_nodes,
          feat_len_y, feat_len_x, ntypes, node_idx);
    }
  }
}

template <typename Idx, typename DType>
__device__ __forceinline__ void Seastar_RgcnLayer1COOKernelEdgePerWarp(
    const Idx* dst_ids, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_nodes, Idx feat_len_y, Idx feat_len_x, Idx ntypes, int edge_idx) {
  Idx dst_id = __ldg(dst_ids + edge_idx);
  Idx src_id = __ldg(src_ids + edge_idx);
  Idx eid = __ldg(eids + edge_idx);
  Idx type_id = __ldg(types + edge_idx);
  Idx lane_idx = threadIdx.x % 32;
  for (int feat_idx = lane_idx; feat_idx < feat_len_x * feat_len_y;
       feat_idx += 32) {
    Idx ty = feat_idx / feat_len_x;
    Idx th = feat_idx % feat_len_x;
    DType agg_val = 0.;
    DType w = 0.;
    Idx cur_type_id = -1;

    if (type_id != cur_type_id) {
      w = __ldg(weight + type_id * feat_len_y * feat_len_x + feat_idx);
    }
    DType n = __ldg(norm + eid);
    DType h = __ldg(hidden + src_id * feat_len_y + ty);

    agg_val += h * w * n;

    atomicAdd(ret + dst_id * feat_len_x + th, agg_val);
  }
}

// NB: in the CSR kernel, the first two parameters are ranges (row_ptr) and
// src_ids (col_idx), which requires transposed graph as input. Now we take in
// dst_ids (row_idx) and src_idx (col_idx) in this COO kernel. To keep
// consistency, we pass row_idx as dst_ids and col_idx as src_ids and use those
// of the transposed graph. This is the same as the CSR kernel.
template <typename Idx, typename DType>
__global__ void HET_Seastar_RgcnLayer1COOKernelImpl(
    const Idx* dst_ids, const Idx* src_ids, const Idx* eids, const Idx* types,
    const DType* hidden, const DType* weight, const DType* norm, DType* ret,
    Idx num_edges, Idx feat_len_y, Idx feat_len_x, Idx ntypes) {
  Idx warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  Idx warp_num = (blockDim.x * gridDim.x) / 32;
  // each warp is assigned consecutive edges
  Idx edge_per_warp = num_edges / warp_num;
  Idx num_warps_with_one_more_edge_assigned = num_edges % warp_num;
  Idx edge_beg;
  if (warp_idx < num_warps_with_one_more_edge_assigned) {
    edge_beg = warp_idx * (edge_per_warp + 1);
  } else {
    edge_beg = warp_idx * edge_per_warp + num_warps_with_one_more_edge_assigned;
  }
  Idx edge_end = edge_beg + edge_per_warp;
  if (warp_idx < num_warps_with_one_more_edge_assigned) {
    edge_end += 1;
  }
  for (Idx edge_idx = edge_beg; edge_idx < edge_end; edge_idx++) {
    // if (warp_idx < num_edges) {
    Seastar_RgcnLayer1COOKernelEdgePerWarp(
        dst_ids, src_ids, eids, types, hidden, weight, norm, ret, num_edges,
        feat_len_y, feat_len_x, ntypes, /*node_idx = */ edge_idx);
  }
}
