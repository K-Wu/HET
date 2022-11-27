#pragma once
#include "DGLHackKernel/DGLHackKernel.h"

template <typename Idx, typename DType>
__global__ void HGTBackwardFusedGradientSmFirstPartGradientAImpl(
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
__global__ void HGTBackwardGradientAImpl(
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
__global__ void HGTBackwardGradientSmFirstPartImpl(
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

template <typename Idx, typename DType, bool UseMuAppliedAttnScoreFlag>
struct BackwardHGTMessageData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      UseMuAppliedAttnScoreFlag || !UseMuAppliedAttnScoreFlag,
      "the program should use partial specialization of this structure");
};

template <typename Idx, typename DType>
struct BackwardHGTMessageData<Idx, DType, true> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_message_src{nullptr}, *edge_softmax_sum{nullptr}, *out{nullptr},
      *grad_out{nullptr};
  DType *unnormalized_attn_score{nullptr}, *mu{nullptr};
  DType* mu_applied_un_softmax_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTMessageData<Idx, DType, false> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_message_src{nullptr}, *edge_softmax_sum{nullptr}, *out{nullptr},
      *grad_out{nullptr};
  DType *unnormalized_attn_score{nullptr}, *mu{nullptr};
  // DType *mu_applied_un_softmax_attn_score{nullptr};
};

// based on _fusedGatBackwardGradFeatSrc, as it is to calculate the gradient of
// message
// TODO: add mu into the term
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag,
          bool UseMuAppliedAttnScoreFlag>
__device__ __forceinline__ void
_hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel(
    BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreFlag> gdata,
    const Idx* row_offsets, const Idx* column_indices, const Idx* etypes,
    int64_t num_rows, const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < num_heads; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        Idx message_src_offset = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx dst_vid_relational = -1;
          Idx etype = -1;
          if constexpr (!CompactAsOfNodeFlag) {
            // in this case, message_src_offset, er_idx and el_idx are related
            // to edge id, regardless of the type of the edge
            message_src_offset = eid * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
              // Idx etype = etypes[e];

              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {  // !ETypeRelPtrFlag
                etype = etypes[e];
              }
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              message_src_offset = src_vid_relational * gdata.message_src_xlen +
                                   head_idx * hidden_xlen + feat_idx;
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
            }
          }

          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?
          Idx sum_vid = dst_vid;
          if constexpr (RelationalFlag && CompactAsOfNodeFlag) {
            sum_vid = dst_vid_relational;
          }

          DType normalized_attn_score;
          if constexpr (FwdOutputMuAppliedAttnScoreFlag) {
            normalized_attn_score =
                gdata.mu_applied_un_softmax_attn_score[eid * num_heads +
                                                       head_idx] /
                gdata.edge_softmax_sum[sum_vid * num_heads + head_idx];
          } else {
            DType mu;
            if constexpr (RelationalFlag) {
              mu = gdata.mu[etype];
            } else {
              mu = gdata.mu[0];
            }
            normalized_attn_score =
                gdata.unnormalized_attn_score[eid * num_heads + head_idx] * mu /
                gdata.edge_softmax_sum[sum_vid * num_heads + head_idx];
          }

          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_message_src + message_src_offset,
                      normalized_attn_score *
                          gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            // exp scheme (both eid and head_idx) could be used for attn_score
            // message_src's could be used for message_src
            s += normalized_attn_score *
                 gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          gdata.grad_message_src[message_src_offset] = s;
        }
      }
    }
  }
}

template <typename Idx, typename DType, bool FwdOutputMuAppliedAttnScoreFlag>
struct BackwardHGTAttnScoreData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      FwdOutputMuAppliedAttnScoreFlag || !FwdOutputMuAppliedAttnScoreFlag,
      "the program should use partial specialization of this structure");
};

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData<Idx, DType, true> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_attn_score{nullptr}, *message_src{nullptr},
      *unnormalized_attn_score{nullptr}, *edge_softmax_sum{nullptr},
      *out{nullptr}, *grad_out{nullptr};
  DType *grad_mu{nullptr}, *mu{nullptr};
  DType* mu_applied_un_softmax_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData<Idx, DType, false> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx* eids;
  DType *grad_attn_score{nullptr}, *message_src{nullptr},
      *unnormalized_attn_score{nullptr}, *edge_softmax_sum{nullptr},
      *out{nullptr}, *grad_out{nullptr};
  DType *grad_mu{nullptr}, *mu{nullptr};
  // DType *mu_applied_un_softmax_attn_score{nullptr};
};

// S_j = expf(z_j) / sum_k expf(z_k)
// deltaz_edge=S_edge*deltaout_dst^T*message_edge - S_edge * deltaout_dst^T *
// out_dst
// TODO: add mu into the term
// based on _fusedGatBackwardGradElEr, as it is to calculate gradient of
// attention
template <typename Idx, typename DType, bool CompactAsOfNodeFlag,
          bool RelationalFlag, bool ETypeRelPtrFlag,
          bool FwdOutputMuAppliedAttnScoreFlag>
__device__ __forceinline__ void _hgtEdgeSoftmaxAccumStageOnlyBackwardKernel(
    BackwardHGTAttnScoreData<Idx, DType, FwdOutputMuAppliedAttnScoreFlag> gdata,
    const Idx* row_offsets, const Idx* column_indices, const Idx* etypes,
    int64_t num_rows, const Idx* unique_srcs_and_dests_rel_ptr,
    const Idx* unique_srcs_and_dests_node_indices, int64_t num_relations) {
  if constexpr (!CompactAsOfNodeFlag) {
    CONSTEXPR_FALSE_CLAUSE_UNREACHABLE(CompactAsOfNodeFlag,
                                       "not implemented yet");
  }
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < num_heads; head_idx += blockDim.x * gridDim.x) {
      for (Idx feat_idx = threadIdx.y; feat_idx < hidden_xlen;
           feat_idx += blockDim.y) {
        DType s = 0.;
        Idx message_src_offset = -1;
        Idx message_src_idx = -1;
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
          message_src_idx =
              (src_vid * num_heads + head_idx) * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx eid = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx edge_softmax_sum_idx = -1;
          Idx dst_vid_relational = -1;
          if constexpr (!CompactAsOfNodeFlag) {
            // in this case, message_src_offset, edge_softmax_sum_idx and
            // message_src_idx are related to edge id, regardless of the type of
            // the edge
            message_src_offset = eid * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
            edge_softmax_sum_idx = eid * num_heads + head_idx;
            message_src_idx =
                (eid * num_heads + head_idx) * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              edge_softmax_sum_idx = dst_vid * num_heads + head_idx;
            } else {
              // in this case, edge_softmax_sum_idx (sum's index) is related to
              // (relation, unique node index) message_src_idx is related to
              // (relation, unique node index) message_src_offset is related to
              // (relation, unique node index) Idx etype = etypes[e];
              Idx etype = -1;
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {
                etype = etypes[e];
              }
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              edge_softmax_sum_idx = dst_vid_relational * num_heads + head_idx;
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              message_src_idx =
                  (src_vid_relational * num_heads + head_idx) * hidden_xlen +
                  feat_idx;
              message_src_offset = src_vid_relational * gdata.message_src_xlen +
                                   head_idx * hidden_xlen + feat_idx;
            }
          }
          Idx dst_out_offset = dst_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;

          DType mu;
          if constexpr (RelationalFlag) {
            mu = gdata.mu[etype];
          } else {
            mu = gdata.mu[0];
          }

          DType normalized_attn_score;
          if constexpr (FwdOutputMuAppliedAttnScoreFlag) {
            normalized_attn_score =
                gdata.mu_applied_un_softmax_attn_score[edge_offset] /
                gdata.edge_softmax_sum[edge_softmax_sum_idx];
          } else {
            normalized_attn_score =
                gdata.unnormalized_attn_score[edge_offset] * mu /
                gdata.edge_softmax_sum[edge_softmax_sum_idx];
          }
          DType grad_for_this_feat_idx =
              gdata.grad_out[dst_out_offset] *
              (gdata.message_src[message_src_offset] -
               gdata.out[dst_out_offset]) *
              normalized_attn_score;
          // el idx scheme could be used for message (only item idx, not the
          // feature idx scheme) exp idx scheme could be used for attn_score
          // if (RelationalFlag){
          //   atomicAdd(gdata.grad_mu + etype,
          //   grad_for_this_feat_idx*gdata.unnormalized_attn_score[edge_offset]);
          // }

          s += grad_for_this_feat_idx * mu;
          if constexpr (!CompactAsOfNodeFlag || RelationalFlag) {
            atomicAdd(gdata.grad_attn_score + edge_offset,
                      grad_for_this_feat_idx * mu);
          }

          atomicAdd(gdata.grad_mu + etype,
                    grad_for_this_feat_idx *
                        gdata.unnormalized_attn_score[edge_offset]);
        }
        if constexpr (CompactAsOfNodeFlag && !RelationalFlag) {
          atomicAdd(gdata.grad_attn_score + edge_offset, s);
        }
      }
    }
  }
}
