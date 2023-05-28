#pragma once

#include <cuda_runtime.h>

#include "kernel_enums.h"

template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
struct BackwardHGTMessageData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      UseMuAppliedAttnScoreSwitch == 0 || !UseMuAppliedAttnScoreSwitch != 0,
      "the program should use partial specialization of this structure");
};

template <typename Idx, typename DType>
struct BackwardHGTMessageData<Idx, DType, 2> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_message_src{nullptr}, *__restrict__ grad_out{
                                                     nullptr};
  DType *__restrict__ normalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTMessageData<Idx, DType, 1> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_message_src{nullptr},
      *__restrict__ grad_out{nullptr},
          *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ unnormalized_attn_score{nullptr}, *__restrict__ mu{
                                                            nullptr};
  DType *__restrict__ mu_softmax_applied_unnormalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTMessageData<Idx, DType, 0> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_message_src{nullptr},
      *__restrict__ grad_out{nullptr},
          *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ unnormalized_attn_score{nullptr}, *__restrict__ mu{
                                                            nullptr};
};

template <typename Idx, typename DType>
struct BackwardNormToUnNormalizedAttnScoreData {
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_normalized_attn_score{nullptr},
      *__restrict__ normalized_attn_score{nullptr},
          *__restrict__ grad_mu{nullptr}, *__restrict__ mu{nullptr};
  DType *__restrict__ unnormalized_attn_score{nullptr},
      *__restrict__ grad_unnormalized_attn_score{nullptr};
};

// NB: no mu flag is used to generalize this scheme to calculate delta q vector
// = delta_attn_score * inner_product
template <typename Idx, typename DType>
struct BackwardToDeltaQData {
  Idx num_heads{0};
  Idx k_vect_dim_per_head{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_unnormalized_attn_score{nullptr},
      *__restrict__ k_inner_product{nullptr},
          *__restrict__ grad_q_vectors{nullptr};
};

// delta_q = delta_attn_score*inner_product
// based on
// HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__global__ void HET__hgtQVectType2BackwardKernel(
    BackwardToDeltaQData<Idx, DType> gdata, const Idx *row_offsets,
    const Idx *column_indices, const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.k_vect_dim_per_head;
  for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y) {
    Idx start_off = row_offsets[dst_vid];
    Idx end_off = row_offsets[dst_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;

        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_idx = gdata.eids[e];
          Idx src_vid = column_indices[e];
          Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                          // !RelationalFlag, the default value is set as 0
          Idx k_inner_product_offset = -1;
          if constexpr (!IsCompact(kind)) {
            // in this case, k_inner_product_offset, er_idx and el_idx are
            // related to edge id, regardless of the type of the edge
            k_inner_product_offset =
                edata_idx * (gdata.k_vect_dim_per_head * num_heads) +
                head_idx * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {  // !ETypeRelPtrFlag
                etype = etypes[e];
              }
              Idx src_vid_relational = find_relational_compact_as_of_node_index(
                  etype, src_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              k_inner_product_offset =
                  src_vid_relational * (gdata.k_vect_dim_per_head * num_heads) +
                  head_idx * hidden_xlen + feat_idx;
            } else {
              // in this case, k_inner_product_offset is the same regardless of
              // which
              // outgoing edge we deal with
              k_inner_product_offset =
                  src_vid * (gdata.k_vect_dim_per_head * num_heads) +
                  head_idx * hidden_xlen + feat_idx;
            }
          }

          // TODO: maybe it's better to cache exp/sum to reduce mem traffic as
          // well as redundant computation?

          DType grad_unnormalized_attn_score =
              gdata.grad_unnormalized_attn_score[edata_idx * num_heads +
                                                 head_idx];

          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_q_vectors +
                          (dst_vid * (gdata.k_vect_dim_per_head * num_heads) +
                           head_idx * hidden_xlen + feat_idx),
                      grad_unnormalized_attn_score *
                          gdata.k_inner_product[k_inner_product_offset]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            // exp scheme (both edata_idx and head_idx) could be used for
            // attn_score message_src's could be used for message_src
            s += grad_unnormalized_attn_score *
                 gdata.k_inner_product[k_inner_product_offset];
          }
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          gdata.grad_q_vectors[(dst_vid *
                                    (gdata.k_vect_dim_per_head * num_heads) +
                                head_idx * hidden_xlen + feat_idx)] = s;
        }
      }
    }
  }
}

// TODO: do an edge parallel version
// based on HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel, as
// normalized_attn_score is present, this is similar to the case where
// FwdOutputMuAppliedAttnScoreSwitch == 2 denote aj as attn_score of edge j,
// mu_j as the corresponding mu of edge j, Sj as the normalized attn score. In
// other words, grad_normalized_attn_score is delta S delta a_j = (-Sum_{i,
// including i==j} Si*Sj*deltaSi + Sj * deltaSj) * mu_j. delta mu_j = (-Sum_{i,
// including i==j} Si*Sj*deltaSi + Sj * deltaSj) * a_j. We first calculate
// -Sum_{incoming edges} (Si * delta Si) at each destination node, and then
// iterate every edge
// NB: Compact as of node is irrelevant here as the data are edge-wise
template <typename Idx, typename DType,  // CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag>
__global__ void HET_EdgeSoftmaxENormToUnNormalizedAttnScoreBackwardKernel(
    BackwardNormToUnNormalizedAttnScoreData<Idx, DType> gdata,
    const Idx *row_offsets, const Idx *column_indices, const Idx *etypes,
    int64_t num_rows, const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  for (Idx src_vid = blockIdx.y * blockDim.y + threadIdx.y; src_vid < num_rows;
       src_vid += gridDim.y * blockDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = blockIdx.x * blockDim.x + threadIdx.x;
         head_idx < num_heads; head_idx += blockDim.x * gridDim.x) {
      DType sum_incoming_edges_product_softmax_score = 0.;
      for (int stage_idx = 0; stage_idx < 2; stage_idx += 1) {
        // stage 1 accumulates the sum of Si * delta Si
        // stage 2 caclulate delta a and delta mu for this edge
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                          // !RelationalFlag, the default value is set as 0

          if constexpr (RelationalFlag) {
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(num_relations, etypes, e);
            } else {
              etype = etypes[e];
            }
          }

          // Compact as of node is irrelevant here as the data are edge-wise

          DType mu = 1.0f;

          if constexpr (RelationalFlag) {
            mu = gdata.mu[etype * num_heads + head_idx];
          } else {
            mu = gdata.mu[head_idx];
          }
          if (stage_idx == 0) {
            sum_incoming_edges_product_softmax_score +=
                gdata.grad_normalized_attn_score[edge_offset] *
                gdata.normalized_attn_score[edge_offset];
          } else {
            // delta a_j = (-Sum_{i, including i==j} Si*Sj*deltaSi + Sj *
            // deltaSj) * mu_j.
            // delta mu_j = (-Sum_{i, including i==j} Si*Sj*deltaSi + Sj *
            // deltaSj) * a_j.
            DType normalized_attn_score =
                gdata.normalized_attn_score[edge_offset];

            DType delta_a_for_curr_edge =
                (-sum_incoming_edges_product_softmax_score +
                 gdata.grad_normalized_attn_score[edge_offset]) *
                normalized_attn_score * mu;
            DType delta_mu_for_curr_edge =
                (-sum_incoming_edges_product_softmax_score +
                 gdata.grad_normalized_attn_score[edge_offset]) *
                normalized_attn_score;

            gdata.grad_unnormalized_attn_score[edge_offset] =
                delta_a_for_curr_edge;

            atomicAdd(gdata.grad_mu + etype * num_heads + head_idx,
                      delta_a_for_curr_edge);
          }
        }
      }
    }
  }
}

// based on _fusedGatBackwardGradFeatSrc, as it is to calculate the gradient of
// message
// NB: notice how mu is involved in the term
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag,
          int UseMuAppliedAttnScoreSwitch>
__global__ void
HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel(
    BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata,
    const Idx *row_offsets, const Idx *column_indices, const Idx *etypes,
    int64_t num_rows, const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        Idx message_src_offset = -1;
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx dst_vid_relational = -1;
          Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                          // !RelationalFlag, the default value is set as 0
          if constexpr (!IsCompact(kind)) {
            // in this case, message_src_offset, er_idx and el_idx are related
            // to edge id, regardless of the type of the edge
            message_src_offset = edata_idx * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (RelationalFlag) {
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

          DType normalized_attn_score;
          if constexpr (UseMuAppliedAttnScoreSwitch == 1) {
            normalized_attn_score =
                gdata.mu_softmax_applied_unnormalized_attn_score[edata_idx *
                                                                     num_heads +
                                                                 head_idx] /
                gdata.edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];
          } else if constexpr (UseMuAppliedAttnScoreSwitch == 2) {
            normalized_attn_score =
                gdata.normalized_attn_score[edata_idx * num_heads + head_idx];
          } else if constexpr (UseMuAppliedAttnScoreSwitch == 0) {
            DType mu;
            if constexpr (RelationalFlag) {
              mu = gdata.mu[etype * num_heads + head_idx];
            } else {
              mu = gdata.mu[head_idx];
            }
            normalized_attn_score =
                gdata
                    .unnormalized_attn_score[edata_idx * num_heads + head_idx] *
                mu /
                gdata.edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];
          }

          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_message_src + message_src_offset,
                      normalized_attn_score *
                          gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {  // CompactAsOfNodeFlag && !RelationalFlag
            // exp scheme (both edata_idx and head_idx) could be used for
            // attn_score message_src's could be used for message_src
            s += normalized_attn_score *
                 gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                head_idx * hidden_xlen + feat_idx];
          }
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          gdata.grad_message_src[message_src_offset] = s;
        }
      }
    }
  }
}

template <typename Idx, typename DType, int FwdOutputMuAppliedAttnScoreSwitch>
struct BackwardHGTAttnScoreData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      FwdOutputMuAppliedAttnScoreSwitch != 0 &&
          FwdOutputMuAppliedAttnScoreSwitch == 0,
      "the program should use partial specialization of this structure");
};

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData<Idx, DType, 2> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_attn_score{nullptr},
      *__restrict__ message_src{nullptr},
          *__restrict__ unnormalized_attn_score{nullptr},
              *__restrict__ out{nullptr}, *__restrict__ grad_out{nullptr};
  DType *__restrict__ grad_mu{nullptr}, *__restrict__ mu{nullptr};
  DType *__restrict__ normalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData<Idx, DType, 1> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_attn_score{nullptr},
      *__restrict__ message_src{nullptr},
          *__restrict__ unnormalized_attn_score{nullptr},
              *__restrict__ out{nullptr}, *__restrict__ grad_out{nullptr};
  DType *__restrict__ grad_mu{nullptr}, *__restrict__ mu{nullptr};
  DType *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ mu_softmax_applied_unnormalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct BackwardHGTAttnScoreData<Idx, DType, 0> {
  Idx num_heads{0};
  Idx message_src_xlen{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ grad_attn_score{nullptr},
      *__restrict__ message_src{nullptr},
          *__restrict__ unnormalized_attn_score{nullptr},
              *__restrict__ out{nullptr}, *__restrict__ grad_out{nullptr};
  DType *__restrict__ grad_mu{nullptr}, *__restrict__ mu{nullptr};
  DType *__restrict__ edgesoftmax_sum_per_node{nullptr};
};

// S_j = expf(z_j) / sum_k expf(z_k)
// deltaz_edge=S_edge*deltaout_dst^T*message_edge - S_edge * deltaout_dst^T *
// out_dst
// NB: notice how mu is involved in the term
// based on _fusedGatBackwardGradElEr, as it is to calculate gradient of
// attention
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag,
          int FwdOutputMuAppliedAttnScoreSwitch>
__global__ void HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel(
    BackwardHGTAttnScoreData<Idx, DType, FwdOutputMuAppliedAttnScoreSwitch>
        gdata,
    const Idx *row_offsets, const Idx *column_indices, const Idx *etypes,
    int64_t num_rows, const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        Idx message_src_offset = -1;
        Idx message_src_idx = -1;
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
          message_src_idx =
              (src_vid * num_heads + head_idx) * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx edgesoftmax_sum_per_node_idx = -1;
          Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                          // !RelationalFlag, the default value is set as 0
          if constexpr (!IsCompact(kind)) {
            // in this case, message_src_offset
            // and message_src_idx are related to edge id, regardless of the
            // type of the edge
            // edgesoftmax_sum_per_node_idx is still one (num_heads,) vector per
            // destination node
            message_src_offset = edata_idx * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
            edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
            message_src_idx =
                (edata_idx * num_heads + head_idx) * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
            } else {
              // in this case, edgesoftmax_sum_per_node_idx (sum's index) is
              // related to (relation, unique node index) message_src_idx is
              // related to (relation, unique node index) message_src_offset is
              // related to (relation, unique node index) Idx etype = etypes[e];

              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {
                etype = etypes[e];
              }
              edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
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
            mu = gdata.mu[etype * num_heads + head_idx];
          } else {
            mu = gdata.mu[head_idx];
          }

          DType normalized_attn_score;
          if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 1) {
            normalized_attn_score =
                gdata.mu_softmax_applied_unnormalized_attn_score[edge_offset] /
                gdata.edgesoftmax_sum_per_node[edgesoftmax_sum_per_node_idx];
          } else if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 0) {
            normalized_attn_score =
                expf(gdata.unnormalized_attn_score[edge_offset] * mu) /
                gdata.edgesoftmax_sum_per_node[edgesoftmax_sum_per_node_idx];
          } else if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 2) {
            normalized_attn_score = gdata.normalized_attn_score[edge_offset];
          } else {
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                FwdOutputMuAppliedAttnScoreSwitch != 0 &&
                    FwdOutputMuAppliedAttnScoreSwitch != 1 &&
                    FwdOutputMuAppliedAttnScoreSwitch != 2,
                "FwdOutputMuAppliedAttnScoreSwitch must be 0, 1 or 2");
          }
          DType grad_for_this_feat_idx =
              gdata.grad_out[dst_out_offset] *
              (gdata.message_src[message_src_offset] -
               gdata.out[dst_out_offset]) *
              normalized_attn_score;
          // el idx scheme could be used for message (only item idx, not the
          // feature idx scheme) exp idx scheme could be used for attn_score

          s += grad_for_this_feat_idx * mu;
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_attn_score + edge_offset,
                      grad_for_this_feat_idx * mu);
          }

          atomicAdd(gdata.grad_mu + etype * num_heads + head_idx,
                    grad_for_this_feat_idx *
                        gdata.unnormalized_attn_score[edge_offset]);
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          atomicAdd(gdata.grad_attn_score + (src_vid * num_heads + head_idx),
                    s);
        }
      }
    }
  }
}

// fusing kernel
// HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel and
// HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel Corresponding python autograd
// function HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR in
// [[hetero_edgesoftmax/python/backend/hgt_layers_and_funcs.py]]
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag,
          int FwdOutputMuAppliedAttnScoreSwitch>
__global__ void HET__hgtAttnAndMessageSrcFusedBckKernel(
    BackwardHGTAttnScoreData<Idx, DType, FwdOutputMuAppliedAttnScoreSwitch>
        gdata,
    DType *grad_message_src, const Idx *row_offsets, const Idx *column_indices,
    const Idx *etypes, int64_t num_rows,
    const Idx *unique_srcs_and_dests_rel_ptr,
    const Idx *unique_srcs_and_dests_node_indices, int64_t num_relations) {
  Idx num_heads = gdata.num_heads;  // originally e_xlen
  Idx hidden_xlen = gdata.message_src_xlen / num_heads;
  for (Idx src_vid = blockIdx.y; src_vid < num_rows; src_vid += gridDim.y) {
    Idx start_off = row_offsets[src_vid];
    Idx end_off = row_offsets[src_vid + 1];
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        DType s_message_src = 0.;
        Idx message_src_offset = -1;
        Idx message_src_idx = -1;
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          // in this case, message_src_offset is the same regardless of which
          // outgoing edge we deal with
          message_src_offset = src_vid * gdata.message_src_xlen +
                               head_idx * hidden_xlen + feat_idx;
          message_src_idx =
              (src_vid * num_heads + head_idx) * hidden_xlen + feat_idx;
        }
        for (Idx e = start_off; e < end_off; ++e) {
          Idx edge_offset = gdata.eids[e] * num_heads + head_idx;
          Idx edata_idx = gdata.eids[e];
          Idx dst_vid = column_indices[e];
          Idx edgesoftmax_sum_per_node_idx = -1;
          Idx dst_vid_relational = -1;
          Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                          // !RelationalFlag, the default value is set as 0
          if constexpr (!IsCompact(kind)) {
            // in this case, message_src_offset
            // and message_src_idx are related to edge id, regardless of the
            // type of the edge
            // edgesoftmax_sum_per_node_idx is still one (num_heads,) vector per
            // destination node
            message_src_offset = edata_idx * gdata.message_src_xlen +
                                 head_idx * hidden_xlen + feat_idx;
            edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
            message_src_idx =
                (edata_idx * num_heads + head_idx) * hidden_xlen + feat_idx;
          } else {  // CompactAsOfNodeFlag
            if constexpr (!RelationalFlag) {
              edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
            } else {
              // in this case, edgesoftmax_sum_per_node_idx (sum's index) is
              // related to (relation, unique node index) message_src_idx is
              // related to (relation, unique node index) message_src_offset is
              // related to (relation, unique node index) Idx etype = etypes[e];

              if constexpr (ETypeRelPtrFlag) {
                etype = binary_search(num_relations, etypes, e);
              } else {
                etype = etypes[e];
              }
              dst_vid_relational = find_relational_compact_as_of_node_index(
                  etype, dst_vid, unique_srcs_and_dests_rel_ptr,
                  unique_srcs_and_dests_node_indices);
              edgesoftmax_sum_per_node_idx = dst_vid * num_heads + head_idx;
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
            mu = gdata.mu[etype * num_heads + head_idx];
          } else {
            mu = gdata.mu[head_idx];
          }

          DType normalized_attn_score;
          if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 1) {
            normalized_attn_score =
                gdata.mu_softmax_applied_unnormalized_attn_score[edge_offset] /
                gdata.edgesoftmax_sum_per_node[edgesoftmax_sum_per_node_idx];
          } else if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 0) {
            normalized_attn_score =
                expf(gdata.unnormalized_attn_score[edge_offset] * mu) /
                gdata.edgesoftmax_sum_per_node[edgesoftmax_sum_per_node_idx];
          } else if constexpr (FwdOutputMuAppliedAttnScoreSwitch == 2) {
            normalized_attn_score = gdata.normalized_attn_score[edge_offset];
          } else {
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                FwdOutputMuAppliedAttnScoreSwitch != 0 &&
                    FwdOutputMuAppliedAttnScoreSwitch != 1 &&
                    FwdOutputMuAppliedAttnScoreSwitch != 2,
                "FwdOutputMuAppliedAttnScoreSwitch must be 0, 1 or 2");
          }
          DType grad_for_this_feat_idx =
              gdata.grad_out[dst_out_offset] *
              (gdata.message_src[message_src_offset] -
               gdata.out[dst_out_offset]) *
              normalized_attn_score;
          // el idx scheme could be used for message (only item idx, not the
          // feature idx scheme) exp idx scheme could be used for attn_score

          s += grad_for_this_feat_idx * mu;
          if constexpr (!IsCompact(kind) || RelationalFlag) {
            atomicAdd(gdata.grad_attn_score + edge_offset,
                      grad_for_this_feat_idx * mu);
            atomicAdd(grad_message_src + message_src_offset,
                      normalized_attn_score *
                          gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                         head_idx * hidden_xlen + feat_idx]);
          } else {
            s_message_src += normalized_attn_score *
                             gdata.grad_out[dst_vid * gdata.message_src_xlen +
                                            head_idx * hidden_xlen + feat_idx];
          }

          atomicAdd(gdata.grad_mu + etype * num_heads + head_idx,
                    grad_for_this_feat_idx *
                        gdata.unnormalized_attn_score[edge_offset]);
        }
        if constexpr (IsCompact(kind) && !RelationalFlag) {
          atomicAdd(gdata.grad_attn_score + (src_vid * num_heads + head_idx),
                    s);
          grad_message_src[message_src_offset] = s_message_src;
        }
      }
    }
  }
}