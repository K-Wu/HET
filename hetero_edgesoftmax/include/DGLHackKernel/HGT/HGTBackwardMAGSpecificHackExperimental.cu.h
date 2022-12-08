#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
__global__ void
HET_HGTBackwardFusedGradientSmFirstPartGradientAImplMAGSpecificHack(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_a,               // |E| * N_HEADS
    DType* grad_sm_first_stage,  //|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
    DType* grad_t_neighbour,     //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,              //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,               //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);
  // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
  int lane_idx = threadIdx.x % 32;
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);
      DType agg0 = 0.;
      DType agg1 = 0.;
      for (; beg < end; beg++) {
        // dealing with each edge
        // broadcast sigma. each thread load one element from m.
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        if (type_id < 2) {
          agg0 += sigma * msg_ele * delta_t_neighbour_ele;
        } else {
          agg1 += sigma * msg_ele * delta_t_neighbour_ele;
        }
        DType inner_agg = sigma * (1 - sigma) * msg_ele * delta_t_neighbour_ele;
// agg += g*w*n;
#pragma unroll
        for (int i_reduction = 16; i_reduction > 0;
             i_reduction = i_reduction / 2) {
          inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
        }

        if (lane_idx == 0) {
          atomicAdd(grad_a + eid * num_heads + tidx_head, inner_agg);
        }
      }
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    0 * num_heads * feat_dim_per_head + tx,
                agg0);
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    1 * num_heads * feat_dim_per_head + tx,
                agg1);
    }
  }
}

template <typename Idx, typename DType>
__global__ void HET_HGTBackwardGradientAImplMAGSpecificHack(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_a,            // |E| * N_HEADS
    DType* grad_t_neighbour,  //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,           //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,            //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);

  // delta a = delta t_neighbour^(l+1) * sigma^-1 * m^T
  int lane_idx = threadIdx.x % 32;
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);

      for (; beg < end; beg++) {
        // dealing with each edge
        // broadcast sigma. each thread load one element from m.
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        DType inner_agg = sigma * (1 - sigma) * msg_ele * delta_t_neighbour_ele;
// agg += g*w*n;
#pragma unroll
        for (int i_reduction = 16; i_reduction > 0;
             i_reduction = i_reduction / 2) {
          inner_agg += __shfl_down_sync(-1, inner_agg, i_reduction);
        }

        if (lane_idx == 0) {
          atomicAdd(grad_a + eid * num_heads + tidx_head, inner_agg);
        }
      }
    }
  }
}

template <typename Idx, typename DType>
__global__ void HET_HGTBackwardGradientSmFirstPartImplMAGSpecificHack(
    Idx* ranges, Idx* dst_ids, Idx* eids, Idx* types,
    DType* grad_sm_first_stage,  //|V| * N_REL_TYPES * N_HEADS * DIM_PER_HEAD
    DType* grad_t_neighbour,     //|V| * N_HEADS * DIM_PER_HEAD
    DType* message,              //|E| * N_HEADS * DIM_PER_HEAD
    DType* sigmas,               //|E| * N_HEADS
    Idx num_nodes, Idx num_heads, Idx feat_dim_per_head, Idx n_rel_types) {
  assert(n_rel_types == 2);  // some bit manipulation is used and thus the
                             // kernel is intended for MAG only

  // delta Sm = \Sum_outgoing (m * delta t_neighbour^(l+1) * sigma)
  // We need to store one delta Sm for each relationship type
  if (blockIdx.x < num_nodes) {
    Idx beg = __ldg(ranges + blockIdx.x);
    Idx end = __ldg(ranges + blockIdx.x + 1);
    Idx tx = threadIdx.x;
    for (; tx < feat_dim_per_head * num_heads; tx += blockDim.x) {
      // each block deald with one source node, and each thread deals with one
      // element along the feature dimension

      Idx tidx_head = tx / feat_dim_per_head;
      Idx tidx_ele_in_head = tx % feat_dim_per_head;
      // load delta t neighbor
      DType delta_t_neighbour_ele =
          __ldg(grad_t_neighbour + blockIdx.x * num_heads * feat_dim_per_head +
                tidx_head * feat_dim_per_head + tidx_ele_in_head);

      // simple hack here effective for OGBN MAG: for grad_sm_first_stage, map
      // relaationship 0, 1 to first half and relationship type 2,3 to the
      // second half, thus reducing 50% footprint
      DType agg0 = 0.;
      DType agg1 = 0.;
      for (; beg < end; beg++) {
        // dealing with each edge
        //  broadcast sigma
        Idx dst_id = __ldg(dst_ids + beg);
        Idx eid = __ldg(eids + beg);
        Idx type_id = __ldg(types + beg);
        DType msg_ele = __ldg(message + eid * num_heads * feat_dim_per_head +
                              tidx_head * feat_dim_per_head + tidx_ele_in_head);
        DType sigma = __ldg(sigmas + eid * num_heads + tidx_head);
        if (type_id < 2) {
          agg0 += sigma * msg_ele * delta_t_neighbour_ele;
        } else {
          agg1 += sigma * msg_ele * delta_t_neighbour_ele;
        }
        // atomicAdd();
      }
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    0 * num_heads * feat_dim_per_head + tx,
                agg0);
      atomicAdd(grad_sm_first_stage +
                    blockIdx.x * n_rel_types * num_heads * feat_dim_per_head +
                    1 * num_heads * feat_dim_per_head + tx,
                agg1);
    }
  }
}
