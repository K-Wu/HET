#pragma once

#include <cuda_runtime.h>

#include "EdgeAttention_4/mysgemm_functor.cu.h"
#include "utils.cu.h"

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
// template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
// __global__ void HET_HGTEdgeMessageConcatenatedCOOKernel(
//     float **intermediate_node_vect, int nnz, int *matCols, int *matRelation,
//     float *node_input_data, float *relation_message_matrices,
//     int **dest_node_to_unique_index_per_relation,
//     int *sizes_unique_index_to_dest_node_per_relation, int num_relations,
//     int *num_blocks_xdim_for_same_relation_per_block_vect,
//     int *beg_node_entry_idxes_vect, int *blockid_relation_id_vect) {
//   constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
//   constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);
//   int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
//   int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
//                COARSE_SGEMM_NODES_PER_BLOCK;
//   int relation_idx = blockid_relation_id_vect[blockIdx.x];

//   for (int node_entry_idx = beg_node_entry_idx;
//        node_entry_idx <
//        sizes_unique_index_to_dest_node_per_relation[relation_idx];
//        node_entry_idx += stride) {
//     mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, false, false,
//                     false, false>::
//         exec_function(
//             OUT_DIM,
//             sizes_unique_index_to_dest_node_per_relation[relation_idx],
//             NODE_INPUT_DIM_PER_HEAD,
//             &relation_message_matrices[relation_idx * NUM_HEADS *
//                                        NODE_INPUT_DIM_PER_HEAD *
//                                        NODE_INPUT_DIM_PER_HEAD],
//             node_input_data, intermediate_node_vect[relation_idx],
//             /*B gather list*/ nullptr, nullptr, -1,
//             /*C scatter list*/ nullptr, node_entry_idx);
//   }
// }

// only requires column indices and therefore both COO and CSR could leverage
// this kernel.
template <typename Idx, typename DType, CompactAsOfNodeKind EdgeMessagesKind,
          bool EdgeMessagesIndirectionOffsetInsteadOf2DArrayFlag,
          typename EdgeMessagesPointerType,
          bool BinarySearchToGetEtypeNodeOffsetFlag, bool CSRInsteadOfCOOFlag>
__device__ __forceinline__ void _HGTTriviallyEdgeParallelNodeMeanAggregation(
    Idx *col_idxes, const ETypeData<Idx, false> etype_data, Idx *eids,
    EdgeMessagesPointerType EdgeMessages, DType *EdgeAttnScores, Idx num_nodes,
    Idx num_edges, Idx num_heads, Idx inout_feat_dim, DType *NodeAggregates,
    Idx *MapAmongEtypeNodeAndOffsetArray, Idx *etype_unique_node_offsets,
    Idx *row_indices_or_row_ptrs) {
  // each warp deals with one edge
  Idx edge_idx =
      (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // warpSize = 32
  assert(inout_feat_dim % warpSize == 0 &&
         "inout_feat_dim must be multiple of warpSize");  // 32
  for (; edge_idx < num_edges; edge_idx += gridDim.x * blockDim.x / warpSize) {
    Idx col_idx = col_idxes[edge_idx];
    // TODO: this looks suspicious
    Idx etype = etype_data.etypes[edge_idx];
    Idx eid = eids[edge_idx];
    DType *EdgeMessage;
    if constexpr (IsCompact(EdgeMessagesKind)) {
      Idx unique_node_index_for_curr_etype;
      Idx row_idx;
      if constexpr (CSRInsteadOfCOOFlag) {
        // do binary search in row_ptrs if CSR
        row_idx = binary_search<Idx, Idx *>(num_edges, row_indices_or_row_ptrs,
                                            edge_idx);
      } else {
        // if COO, just index the row_idx directly
        row_idx = row_indices_or_row_ptrs[edge_idx];
      }
      if constexpr (BinarySearchToGetEtypeNodeOffsetFlag) {
        unique_node_index_for_curr_etype = binary_search<Idx, Idx *>(
            etype_unique_node_offsets[etype + 1] -
                etype_unique_node_offsets[etype],
            &MapAmongEtypeNodeAndOffsetArray[etype_unique_node_offsets[etype]],
            row_idx);
      } else {
        unique_node_index_for_curr_etype =
            MapAmongEtypeNodeAndOffsetArray[num_nodes * etype + row_idx];
      }
      if constexpr (EdgeMessagesIndirectionOffsetInsteadOf2DArrayFlag) {
        EdgeMessage = &EdgeMessages[etype_unique_node_offsets[etype] +
                                    unique_node_index_for_curr_etype];
      } else {
        EdgeMessage = &EdgeMessages[etype][unique_node_index_for_curr_etype];
      }

    } else {
      EdgeMessage = &EdgeMessages[eid * inout_feat_dim];
    }
    DType *EdgeAttnScore = &EdgeAttnScores[eid * num_heads];
    Idx featid = threadIdx.x % warpSize;

    for (; featid < inout_feat_dim; featid += warpSize) {
      Idx head_id = featid / (inout_feat_dim / num_heads);
      // NodeAggregates should be initialized to 0 before this kernel launch
      atomicAdd(&NodeAggregates[col_idx * inout_feat_dim + featid],
                EdgeMessage[head_id * (inout_feat_dim / num_heads) +
                            featid % (inout_feat_dim / num_heads)] *
                    EdgeAttnScore[head_id]);
    }
  }
}

// We need to use separate coo at this moment
// this kernel can be used for either sW in edge attention computation (vanilla
// + compactAsOfNode) or sW in edge message generation (vanilla). s STANDS FOR
// SOURCE NODE.
// template <typename Idx, typename DType, int TILE_SZ_A, int TILE_SZ_B,
//           int OUT_DIM, int NUM_HEADS, bool WORK_ASSIGNMENT_INDEX_FLAG,
//           bool InputNodeFeaturesCompactOfNodeFlag,
//           bool ETypeUniqueNodeIndexBinarySearchFlag>
// __global__ void HET_EdgeMessageGeneration(
//     Idx *etype_edge_offsets, const ETypeData<Idx, true> etype_data, Idx
//     *row_idxes, Idx *col_idxes,  Idx *eids, DType *weight, Idx num_nodes, Idx
//     num_edges, DType *NodeFeatures, DType *OutEdgeMessage, Idx
//     *etype_unique_node_offsets, Idx *etype_unique_node_index_map) {
//   constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
//   constexpr int COARSE_SGEMM_EDGES_PER_BLOCK = (TILE_SZ_B);

//   int stride;
//   int relation_idx;
//   if constexpr (WORK_ASSIGNMENT_INDEX_FLAG) {
//     assert(0 &&
//            "WORK_ASSIGNMENT_INDEX_FLAG is not supported in "
//            "HET_EdgeMessageGeneration");
//   } else {
//     relation_idx =
//         binary_search<Idx, Idx *>(etype_data.num_relations,
//         etype_data.etypes, blockIdx.x);
//     stride =
//         COARSE_SGEMM_EDGES_PER_BLOCK * (etype_data.etypes[relation_idx + 1] -
//                                         etype_data.etypes[relation_idx]);
//   }
//   for (int edge_entry_idx = etype_edge_offsets[relation_idx];
//        edge_entry_idx < etype_edge_offsets[relation_idx + 1];
//        edge_entry_idx += stride) {
//     if constexpr (InputNodeFeaturesCompactOfNodeFlag) {
//       mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, true,
//                       InputNodeFeaturesCompactOfNodeFlag,
//                       ETypeUniqueNodeIndexBinarySearchFlag>::
//           exec_function(
//               OUT_DIM,
//               etype_edge_offsets[relation_idx + 1] -
//                   etype_edge_offsets[relation_idx],
//               NODE_INPUT_DIM_PER_HEAD,
//               &weight[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
//                       NODE_INPUT_DIM_PER_HEAD],
//               // assuming in_dim == out_dim
//               &NodeFeatures[etype_unique_node_offsets[relation_idx] *
//               OUT_DIM], OutEdgeMessage,
//               &row_idxes[etype_edge_offsets[relation_idx]] /* source node
//                            feature is what we want as one of the operand*/
//               ,
//               &etype_unique_node_index_map
//                   [etype_unique_node_offsets[relation_idx]],
//               etype_unique_node_offsets[relation_idx + 1] -
//                   etype_unique_node_offsets[relation_idx],
//               &eids[etype_edge_offsets[relation_idx]],
//               edge_entry_idx - etype_edge_offsets[relation_idx]);
//     } else {
//       mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, true,
//                       false, false>::
//           exec_function(
//               OUT_DIM, etype_edge_offsets[relation_idx + 1],
//               NODE_INPUT_DIM_PER_HEAD,
//               &weight[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
//                       NODE_INPUT_DIM_PER_HEAD],
//               // assuming in_dim == out_dim
//               NodeFeatures, OutEdgeMessage,
//               row_idxes /* source node feature is what we want as one of the
//                            operand*/
//               ,
//               nullptr, -1, eids, edge_entry_idx);
//     }
//   }
// }

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
// constexpr auto HGTVanillaEdgeAttentionSecondStage =
//     HET_GeneralEdgeMessageMultiplyNodeFeature<
//         float, /*ProductCompactAsOfNodeFlag = */
//         CompactAsOfNodeKind::Disabled,
//         /*EidEnableFlag = */ true, float *>;

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
// constexpr auto HGTCompactAsOfNodesEdgeAttentionSecondStage =
//     HET_GeneralEdgeMessageMultiplyNodeFeature<float, true, false, float **>;

template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
struct HgtDstOutData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      UseMuAppliedAttnScoreSwitch != 0 || UseMuAppliedAttnScoreSwitch == 0,
      "the program should use partial specialization of this structure");
};

// TODO: use designated intializer, as explained in
// https://en.cppreference.com/w/cpp/language/aggregate_initialization
template <typename Idx, typename DType>
struct HgtDstOutData<Idx, DType, 0> {
  Idx num_heads{0};
  Idx message_out_dim{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ edgesoftmax_sum_per_node{nullptr},
      *__restrict__ message{nullptr}, *__restrict__ ret{nullptr};
  DType *__restrict__ mu{nullptr},
      *__restrict__ unnormalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct HgtDstOutData<Idx, DType, 1> {
  Idx num_heads{0};
  Idx message_out_dim{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ edgesoftmax_sum_per_node{nullptr},
      *__restrict__ message{nullptr}, *__restrict__ ret{nullptr};
  DType *__restrict__ mu_softmax_applied_unnormalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct HgtDstOutData<Idx, DType, 2> {
  Idx num_heads{0};
  Idx message_out_dim{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ message{nullptr}, *__restrict__ ret{nullptr};
  DType *__restrict__ normalized_attn_score{nullptr};
};

// based on _gatSumProdZipDivKernel originally from seastar dgl-hack
// src/kernel/cuda/binary_reduce_impl.cu This kernel calculates attn_score based
// on unnormalized attn_score and edge_sfotmax sum at each destination nodes,
// and apply it in the fly to each edge message, and finally accumulates the
// result to the destination node.
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag,
          int UseMuAppliedAttnScoreSwitch>
__global__ void HET_HGTMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum(
    HgtDstOutData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata,
    const Idx *row_offsets, const Idx *column_indices,
    const ETypeData<Idx, ETypeRelPtrFlag> etype_data, int64_t num_rows,
    const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx num_heads = gdata.num_heads;
  Idx hidden_xlen = gdata.message_out_dim / num_heads;
  for (Idx dst_vid = blockIdx.y; dst_vid < num_rows; dst_vid += gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);
    for (Idx head_idx = threadIdx.y; head_idx < num_heads;
         head_idx += blockDim.y) {
      for (Idx feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
           feat_idx < hidden_xlen; feat_idx += blockDim.x * gridDim.x) {
        DType s = 0.;
        for (Idx eidx = start_off; eidx < end_off; eidx++) {
          Idx src_vid = column_indices[eidx];
          Idx feat_src_entry_id = -1;
          Idx edata_idx = gdata.eids[eidx];
          if constexpr (RelationalFlag) {
            Idx etype = -1;
            if constexpr (ETypeRelPtrFlag) {
              etype = binary_search(etype_data.num_relations, etype_data.etypes,
                                    eidx);
            } else {
              etype = etype_data.etypes[eidx];
            }
            if constexpr (IsCompact(kind)) {
              // TODO: etype is not needed if etype_mapper_data
              // !IsBinarySearch(kind)
              feat_src_entry_id = find_relational_compact_as_of_node_index(
                  etype, src_vid, edata_idx, etype_mapper_data);

            } else {
              // NB: we need to use edata_idx instead of eidx here
              feat_src_entry_id = edata_idx;
            }

            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing along
              // the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  FullCartesianFlag, "should be non-reachable not implemented");
            }
            // NB: e_xlen is the number of heads, feat_src_xlen is
            // message_out_dim, hidden_xlen is message_out_dim//num_heads

            DType normalized_attn_score;
            if constexpr (UseMuAppliedAttnScoreSwitch == 0) {
              normalized_attn_score =
                  expf(gdata.mu[etype * num_heads + head_idx] *
                       gdata.unnormalized_attn_score[edata_idx * num_heads +
                                                     head_idx]) /
                  gdata
                      .edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];
            } else if constexpr (UseMuAppliedAttnScoreSwitch == 1) {
              normalized_attn_score =
                  gdata.mu_softmax_applied_unnormalized_attn_score
                      [edata_idx * num_heads + head_idx] /
                  gdata
                      .edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];
            } else if constexpr (UseMuAppliedAttnScoreSwitch == 2) {
              normalized_attn_score =
                  gdata.normalized_attn_score[edata_idx * num_heads + head_idx];
            } else {
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  UseMuAppliedAttnScoreSwitch != 0 &&
                      UseMuAppliedAttnScoreSwitch != 1 &&
                      UseMuAppliedAttnScoreSwitch != 2,
                  "should be non-reachable");
            }
            s += (normalized_attn_score *
                  gdata.message[feat_src_entry_id * gdata.message_out_dim +
                                head_idx * hidden_xlen + feat_idx]);
          } else {  // !RelationalFlag
            // NB: feat_src_entry_id varies between edata_idx and src_vid
            // depending on compactasofnodeflag
            if constexpr (IsCompact(kind)) {
              feat_src_entry_id = src_vid;
            } else {
              feat_src_entry_id = edata_idx;
            }
            DType normalized_attn_score;
            // TODO: extend UseMuAppliedAttnScoreFlag to a switch that could
            // ouptut softmaxed attn score
            if constexpr (UseMuAppliedAttnScoreSwitch == 0) {
              normalized_attn_score =
                  gdata.mu[head_idx] *
                  gdata.unnormalized_attn_score[edata_idx * num_heads +
                                                head_idx] /
                  gdata
                      .edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];
            } else if constexpr (UseMuAppliedAttnScoreSwitch == 1) {
              normalized_attn_score =
                  gdata.mu_softmax_applied_unnormalized_attn_score
                      [edata_idx * num_heads + head_idx] /
                  gdata
                      .edgesoftmax_sum_per_node[dst_vid * num_heads + head_idx];

            } else if constexpr (UseMuAppliedAttnScoreSwitch == 2) {
              normalized_attn_score =
                  gdata.normalized_attn_score[edata_idx * num_heads + head_idx];
            } else {
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  UseMuAppliedAttnScoreSwitch != 0 &&
                      UseMuAppliedAttnScoreSwitch != 1 &&
                      UseMuAppliedAttnScoreSwitch != 2,
                  "should be non-reachable");
            }
            s += normalized_attn_score *
                 gdata.message[feat_src_entry_id * gdata.message_out_dim +
                               head_idx * hidden_xlen + feat_idx];
          }
        }

        gdata.ret[dst_vid * gdata.message_out_dim + head_idx * hidden_xlen +
                  feat_idx] = s;
      }
    }
  }
}

template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch>
struct HgtEdgeSoftmaxAccumData {
  CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
      OutputMuAppliedAttnScoreSwitch != 0 ||
          OutputMuAppliedAttnScoreSwitch == 0,
      "the program should use partial specialization of this structure");
};

template <typename Idx, typename DType>
struct HgtEdgeSoftmaxAccumData<Idx, DType, 0> {
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ mu{nullptr},
      *__restrict__ unnormalized_attn_score{nullptr},
      *__restrict__ edgesoftmax_sum_per_node{nullptr};
};

template <typename Idx, typename DType>
struct HgtEdgeSoftmaxAccumData<Idx, DType, 1> {
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ mu{nullptr},
      *__restrict__ unnormalized_attn_score{nullptr},
      *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ mu_softmax_applied_unnormalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct HgtEdgeSoftmaxAccumData<Idx, DType, 2> {
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ mu{nullptr},
      *__restrict__ unnormalized_attn_score{nullptr},
      *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ normalized_attn_score{nullptr};
};

template <typename Idx, typename DType>
struct HgtEdgeSoftmaxAccumData<Idx, DType, 3> {
  Idx num_heads{0};
  Idx *__restrict__ eids{nullptr};
  DType *__restrict__ mu{nullptr},
      *__restrict__ unnormalized_attn_score{nullptr},
      *__restrict__ edgesoftmax_sum_per_node{nullptr};
  DType *__restrict__ mu_softmax_applied_unnormalized_attn_score{nullptr};
  DType *__restrict__ normalized_attn_score{nullptr};
};

// TODO: add mu
// based on _gatExpLeakyReluSumKernel originally from seastar dgl-hack
// src/kernel/cuda/binary_reduce_impl.cu this function only calculates edge
// softmax sum at each destination node, where the edge softmax normalization
// of attention was expected to not only do such accumulation, but also use it
// as the devisor to normalize each edge.
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag,
          int OutputMuAppliedAttnScoreSwitch>
__global__ void HET_HGTEdgeSoftmaxAccumStageOnlyKernel(
    HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata,
    const Idx *row_offsets, const Idx *column_indices,
    const ETypeData<Idx, ETypeRelPtrFlag> etype_data, int64_t num_rows,
    const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  if constexpr (OutputMuAppliedAttnScoreSwitch != 0 &&
                OutputMuAppliedAttnScoreSwitch != 1 &&
                OutputMuAppliedAttnScoreSwitch != 2 &&
                OutputMuAppliedAttnScoreSwitch != 3) {
    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
        OutputMuAppliedAttnScoreSwitch != 0 &&
            OutputMuAppliedAttnScoreSwitch != 1 &&
            OutputMuAppliedAttnScoreSwitch != 2 &&
            OutputMuAppliedAttnScoreSwitch != 3,
        "the program should use partial specialization of this "
        "structure");
  }
  Idx num_heads = gdata.num_heads;
  for (Idx dst_vid = ty; dst_vid < num_rows;
       dst_vid += blockDim.y * gridDim.y) {
    Idx start_off = *(row_offsets + dst_vid);
    Idx end_off = *(row_offsets + dst_vid + 1);

    for (Idx feat_idx = tx; feat_idx < num_heads;
         feat_idx += blockDim.x * gridDim.x) {
      DType sum = 0.;
      for (Idx eidx = start_off; eidx < end_off; ++eidx) {
        Idx src_id = *(column_indices + eidx);
        Idx feat_off_src = -1;
        Idx edata_idx = gdata.eids[eidx];
        Idx etype = -1;
        DType mu = 1.0;
        if constexpr (RelationalFlag) {
          if constexpr (ETypeRelPtrFlag) {
            etype = binary_search(etype_data.num_relations, etype_data.etypes,
                                  eidx);
          } else {
            etype = etype_data.etypes[eidx];
          }
        }
        if constexpr (IsCompact(kind)) {
          if constexpr (RelationalFlag) {
            if constexpr (FullCartesianFlag) {
              // NB: This is the case where we have the data stored in
              // (relation, node) but do not compress the (relation, node)
              // matrix. It could be a case in subgraph where compressing
              // along the node dimension may not be worth it.
              CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                  IsCompact(kind) && RelationalFlag && FullCartesianFlag,
                  "should be non-reachable not implemented");
            }
            Idx src_vid_relational = find_relational_compact_as_of_node_index(
                etype, src_id, edata_idx, etype_mapper_data);
            feat_off_src = src_vid_relational * num_heads + feat_idx;
          } else {
            feat_off_src = src_id * num_heads + feat_idx;
          }
        } else {
          // per edge
          feat_off_src = edata_idx * num_heads + feat_idx;
        }
        if constexpr (RelationalFlag) {
          mu = gdata.mu[etype * num_heads + feat_idx];
        } else {
          mu = gdata.mu[feat_idx];
        }

        // TODO: we need to determine where to calculate expf as the
        // non-linearity of edgesoftmax
        DType tmp = expf(gdata.unnormalized_attn_score[feat_off_src] * mu);
        // NB: e_xlen is num_heads
        if constexpr (RelationalFlag) {
          // NB: double check dst_vid_relational is defined when
          // !CompactAsOfNodeFlag && RelationalFlag
          // TODO: fix this and align dst_vid_relational definition with
          // _fusedGatBackwardGradElErFeatSrcFused
          atomicAdd(&gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
                                                    feat_idx],
                    tmp);
        }
        if constexpr (OutputMuAppliedAttnScoreSwitch == 1 ||
                      OutputMuAppliedAttnScoreSwitch == 3) {
          gdata.mu_softmax_applied_unnormalized_attn_score[feat_off_src] = tmp;
        }

        sum += tmp;
      }
      if constexpr (!RelationalFlag) {
        gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) + feat_idx] =
            sum;
      }

      if constexpr (OutputMuAppliedAttnScoreSwitch == 2 ||
                    OutputMuAppliedAttnScoreSwitch == 3) {
        for (Idx feat_idx = tx; feat_idx < num_heads;
             feat_idx += blockDim.x * gridDim.x) {
          for (Idx eidx = start_off; eidx < end_off; ++eidx) {
            Idx src_id = *(column_indices + eidx);
            Idx edata_idx = gdata.eids[eidx];
            if constexpr (OutputMuAppliedAttnScoreSwitch == 3) {
              gdata.normalized_attn_score[edata_idx * num_heads + feat_idx] =
                  gdata.mu_softmax_applied_unnormalized_attn_score
                      [edata_idx * num_heads + feat_idx] /
                  gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
                                                 feat_idx];
            } else {
              DType mu;
              // fixed the feat_off_src undefined bug
              // TODO: fix similar bugs in the other kernels
              Idx feat_off_src = -1;
              // TODO: etype == -1 only for passing the compilation, needs to
              // define it
              Idx etype = -1;
              if constexpr (IsCompact(kind)) {
                if constexpr (RelationalFlag) {
                  if constexpr (FullCartesianFlag) {
                    // NB: This is the case where we have the data stored in
                    // (relation, node) but do not compress the (relation, node)
                    // matrix. It could be a case in subgraph where compressing
                    // along the node dimension may not be worth it.
                    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                        IsCompact(kind) && RelationalFlag && FullCartesianFlag,
                        "should be non-reachable not implemented");
                  }
                  Idx src_vid_relational =
                      find_relational_compact_as_of_node_index(
                          etype, src_id, edata_idx, etype_mapper_data);
                  feat_off_src = src_vid_relational * num_heads + feat_idx;
                } else {
                  feat_off_src = src_id * num_heads + feat_idx;
                }
              } else {
                // per edge
                feat_off_src = edata_idx * num_heads + feat_idx;
              }
              if constexpr (RelationalFlag) {
                Idx etype;
                if constexpr (ETypeRelPtrFlag) {
                  etype = binary_search(etype_data.num_relations,
                                        etype_data.etypes, eidx);
                } else {
                  etype = etype_data.etypes[eidx];
                }
                mu = gdata.mu[etype * num_heads + feat_idx];
              } else {
                mu = gdata.mu[feat_idx];
              }
              gdata.normalized_attn_score[edata_idx * num_heads + feat_idx] =
                  gdata.unnormalized_attn_score[feat_off_src] * mu /
                  gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
                                                 feat_idx];
            }
          }
        }
      }
    }
  }
}

// head -> blockIdx.x * blockDim.x + threadIdx.x;
// edge -> blockIdx.y * blockDim.y + threadIdx.y;
template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag,
          int OutputMuAppliedAttnScoreSwitch>
__global__ void HET_HGTEdgeSoftmaxAccumStageOnlyKernel_edgeparallel(
    HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata,
    const Idx *row_indices, const Idx *column_indices,
    const ETypeData<Idx, ETypeRelPtrFlag> etype_data, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data) {
  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;

  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;

  if constexpr (OutputMuAppliedAttnScoreSwitch != 0 &&
                OutputMuAppliedAttnScoreSwitch != 1 &&
                OutputMuAppliedAttnScoreSwitch != 2 &&
                OutputMuAppliedAttnScoreSwitch != 3) {
    CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
        OutputMuAppliedAttnScoreSwitch != 0 &&
            OutputMuAppliedAttnScoreSwitch != 1 &&
            OutputMuAppliedAttnScoreSwitch != 2 &&
            OutputMuAppliedAttnScoreSwitch != 3,
        "the program should use partial specialization of this "
        "structure");
  }
  Idx num_heads = gdata.num_heads;
  for (Idx eidx = ty; eidx < num_edges; eidx += blockDim.y * gridDim.y) {
    Idx dst_vid = *(row_indices + eidx);

    for (Idx feat_idx = tx; feat_idx < num_heads;
         feat_idx += blockDim.x * gridDim.x) {
      DType sum = 0.;
      Idx src_id = *(column_indices + eidx);
      Idx feat_off_src = -1;
      Idx edata_idx = gdata.eids[eidx];
      Idx etype = 0;  // NB: as mu needs to refer to etype even in case of
                      // !RelationalFlag, the default value is set as 0
      DType mu;
      if constexpr (RelationalFlag) {
        if constexpr (ETypeRelPtrFlag) {
          if constexpr (EtypeRelPtrIndexSearch) {
            etype = linear_search(etype_data.num_relations, etype_data.etypes,
                                  eidx, resume_from);
            resume_from = etype;
          } else {
            etype = binary_search(etype_data.num_relations, etype_data.etypes,
                                  eidx);
          }

        } else {
          etype = etype_data.etypes[eidx];
        }
      }
      if constexpr (IsCompact(kind)) {
        if constexpr (RelationalFlag) {
          if constexpr (FullCartesianFlag) {
            // NB: This is the case where we have the data stored in
            // (relation, node) but do not compress the (relation, node)
            // matrix. It could be a case in subgraph where compressing
            // along the node dimension may not be worth it.
            CONSTEXPR_TRUE_CLAUSE_UNREACHABLE(
                IsCompact(kind) && RelationalFlag && FullCartesianFlag,
                "should be non-reachable not implemented");
          }
          Idx src_vid_relational = find_relational_compact_as_of_node_index(
              etype, src_id, edata_idx, etype_mapper_data);
          feat_off_src = src_vid_relational * num_heads + feat_idx;
        } else {
          feat_off_src = src_id * num_heads + feat_idx;
        }
      } else {
        // per edge
        feat_off_src = edata_idx * num_heads + feat_idx;
      }
      if constexpr (RelationalFlag) {
        mu = gdata.mu[etype * num_heads + feat_idx];
      } else {
        mu = gdata.mu[feat_idx];
      }
      // TODO: we need to determine where to calculate expf as the
      // non-linearity of edgesoftmax
      DType tmp = expf(gdata.unnormalized_attn_score[feat_off_src] * mu);
      // NB: e_xlen is num_heads
      if constexpr (RelationalFlag) {
        // NB: double check dst_vid_relational is defined when
        // !CompactAsOfNodeFlag && RelationalFlag
        // TODO: fix this and align dst_vid_relational definition with
        // _fusedGatBackwardGradElErFeatSrcFused
        atomicAdd(
            &gdata
                 .edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) + feat_idx],
            tmp);
      }
      if constexpr (OutputMuAppliedAttnScoreSwitch == 1 ||
                    OutputMuAppliedAttnScoreSwitch == 3) {
        gdata.mu_softmax_applied_unnormalized_attn_score[feat_off_src] = tmp;
      }

      sum += tmp;
      //}
      if constexpr (!RelationalFlag) {
        atomicAdd(
            &gdata
                 .edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) + feat_idx],
            sum);
      }
    }
  }
}

template <typename Idx, typename DType, CompactAsOfNodeKind kind,
          bool RelationalFlag, bool ETypeRelPtrFlag, bool FullCartesianFlag,
          int OutputMuAppliedAttnScoreSwitch>
__global__ void HET_HGTEdgeSoftmaxAccumStageOnlyKernel_edgeparallel_stage_2(
    HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata,
    const Idx *row_indices, const Idx *column_indices,
    const ETypeData<Idx, ETypeRelPtrFlag> etype_data, int64_t num_edges,
    const ETypeMapperData<Idx, kind> etype_mapper_data) {
  constexpr bool EtypeRelPtrIndexSearch = true;
  Idx resume_from = 0;

  Idx tx = blockIdx.x * blockDim.x + threadIdx.x;
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx num_heads = gdata.num_heads;
  if constexpr (OutputMuAppliedAttnScoreSwitch == 2 ||
                OutputMuAppliedAttnScoreSwitch == 3) {
    for (Idx eidx = ty; eidx < num_edges; eidx += blockDim.y * gridDim.y) {
      Idx src_id = *(column_indices + eidx);

      Idx edata_idx = gdata.eids[eidx];
      Idx dst_vid = *(row_indices + eidx);
      for (Idx feat_idx = tx; feat_idx < num_heads;
           feat_idx += blockDim.x * gridDim.x) {
        Idx feat_off_src = -1;
        if constexpr (OutputMuAppliedAttnScoreSwitch == 3) {
          gdata.normalized_attn_score[edata_idx * num_heads + feat_idx] =
              gdata.mu_softmax_applied_unnormalized_attn_score[edata_idx *
                                                                   num_heads +
                                                               feat_idx] /
              gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
                                             feat_idx];
        } else {
          DType mu;
          if constexpr (RelationalFlag) {
            Idx etype;
            if constexpr (ETypeRelPtrFlag) {
              if constexpr (EtypeRelPtrIndexSearch) {
                etype = linear_search(etype_data.num_relations,
                                      etype_data.etypes, eidx, resume_from);
                resume_from = etype;
              } else {
                etype = binary_search(etype_data.num_relations,
                                      etype_data.etypes, eidx);
              }
            } else {
              etype = etype_data.etypes[eidx];
            }
            mu = gdata.mu[etype * num_heads + feat_idx];
          } else {
            mu = gdata.mu[feat_idx];
          }
          gdata.normalized_attn_score[edata_idx * num_heads + feat_idx] =
              gdata.unnormalized_attn_score[feat_off_src] * mu /
              gdata.edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
                                             feat_idx];
        }
      }
    }
  }
}
