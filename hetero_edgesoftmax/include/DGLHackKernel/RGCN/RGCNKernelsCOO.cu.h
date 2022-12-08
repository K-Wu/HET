#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
__device__ __forceinline__ void RgcnLayer1COOKernelEdgePerWarp(
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
__global__ void HET_RgcnLayer1COOKernelImpl(
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
    RgcnLayer1COOKernelEdgePerWarp(
        dst_ids, src_ids, eids, types, hidden, weight, norm, ret, num_edges,
        feat_len_y, feat_len_x, ntypes, /*node_idx = */ edge_idx);
  }
}
