#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGT/HGTPreprocessing.h"
#include "EdgeAttention_4/EdgeAttentionCOO.h"
#include "EdgeAttention_4/mysgemm_functor.cu.h"
#include "utils.cu.h"

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
__global__ void _global_EdgeMessageConcatenatedCOOKernel(
    float** intermediate_node_vect, int nnz, int* matCols, int* matRelation,
    float* node_input_data, float* relation_message_matrices,
    int** dest_node_to_unique_index_per_relation,
    int* sizes_unique_index_to_dest_node_per_relation, int num_relations,
    int* num_blocks_xdim_for_same_relation_per_block_vect,
    int* beg_node_entry_idxes_vect, int* blockid_relation_id_vect) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);
  int beg_node_entry_idx = beg_node_entry_idxes_vect[blockIdx.x];
  int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
               COARSE_SGEMM_NODES_PER_BLOCK;
  int relation_idx = blockid_relation_id_vect[blockIdx.x];

  for (int node_entry_idx = beg_node_entry_idx;
       node_entry_idx <
       sizes_unique_index_to_dest_node_per_relation[relation_idx];
       node_entry_idx += stride) {
    mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, false, false,
                    false, false>::
        exec_function(
            OUT_DIM, sizes_unique_index_to_dest_node_per_relation[relation_idx],
            NODE_INPUT_DIM_PER_HEAD,
            &relation_message_matrices[relation_idx * NUM_HEADS *
                                       NODE_INPUT_DIM_PER_HEAD *
                                       NODE_INPUT_DIM_PER_HEAD],
            node_input_data, intermediate_node_vect[relation_idx],
            /*B gather list*/ nullptr, nullptr, -1,
            /*C scatter list*/ nullptr, node_entry_idx);
  }
}

// only requires column indices and therefore both COO and CSR could leverage
// this kernel.
template <typename Idx, typename DType, bool EdgeMessagesCompactAsOfNodeFlag,
          bool EdgeMessagesIndirectionOffsetInsteadOf2DArrayFlag,
          typename EdgeMessagesPointerType,
          bool BinarySearchToGetEtypeNodeOffsetFlag, bool CSRInsteadOfCOOFlag>
__device__ __forceinline__ void _HGTTriviallyEdgeParallelNodeMeanAggregation(
    Idx* col_idxes, Idx* etypes, Idx* eids,
    EdgeMessagesPointerType EdgeMessages, DType* EdgeAttnScores, Idx num_nodes,
    Idx num_edges, Idx num_etypes, Idx num_heads, Idx inout_feat_dim,
    DType* NodeAggregates, Idx* MapAmongEtypeNodeAndOffsetArray,
    Idx* etype_unique_node_offsets, Idx* row_indices_or_row_ptrs) {
  // each warp deals with one edge
  Idx edge_idx =
      (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // warpSize = 32
  assert(inout_feat_dim % warpSize == 0 &&
         "inout_feat_dim must be multiple of warpSize");  // 32
  for (; edge_idx < num_edges; edge_idx += gridDim.x * blockDim.x / warpSize) {
    Idx col_idx = col_idxes[edge_idx];
    Idx etype = etypes[edge_idx];
    Idx eid = eids[edge_idx];
    DType* EdgeMessage;
    if constexpr (EdgeMessagesCompactAsOfNodeFlag) {
      Idx unique_node_index_for_curr_etype;
      Idx row_idx;
      if constexpr (CSRInsteadOfCOOFlag) {
        // do binary search in row_ptrs if CSR
        row_idx = binary_search<Idx, Idx*>(num_edges, row_indices_or_row_ptrs,
                                           edge_idx);
      } else {
        // if COO, just index the row_idx directly
        row_idx = row_indices_or_row_ptrs[edge_idx];
      }  // if constexpr (CSRInsteadOfCOOFlag) {
      if constexpr (BinarySearchToGetEtypeNodeOffsetFlag) {
        unique_node_index_for_curr_etype = binary_search<Idx, Idx*>(
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
    DType* EdgeAttnScore = &EdgeAttnScores[eid * num_heads];
    Idx featid = threadIdx.x % warpSize;

    for (; featid < inout_feat_dim; featid += warpSize) {
      Idx head_id = featid / (inout_feat_dim /
                              num_heads);  // featid / NODE_INPUT_DIM_PER_HEAD;
      // NodeAggregates should be initialized to 0 before this kernel launch
      atomicAdd(
          &NodeAggregates[col_idx * inout_feat_dim + featid],
          EdgeMessage[head_id * (inout_feat_dim / num_heads) +
                      featid % (inout_feat_dim / num_heads)] *
              EdgeAttnScore[head_id]);  // featid % NODE_INPUT_DIM_PER_HEAD];
    }
  }
}

__global__ void HGTTriviallyEdgeParallelVanillaNodeMeanAggregation(
    int64_t* col_idxes, int64_t* etypes, int64_t* eids, float* EdgeMessages,
    float* EdgeAttnScores, int64_t num_nodes, int64_t num_edges,
    int64_t num_etypes, int64_t num_heads, int64_t inout_feat_dim,
    float* NodeAggregates) {
  _HGTTriviallyEdgeParallelNodeMeanAggregation<
      int64_t, float, /* EdgeMessagesCompactAsOfNodeFlag = */ false,
      /* EdgeMessagesIndirectionOffsetInsteadOf2DArrayFlag = */ false, float*,
      /*flag not applicable*/ false, /*flag not applicable*/ false>(
      col_idxes, etypes, eids, EdgeMessages, EdgeAttnScores, num_nodes,
      num_edges, num_etypes, num_heads, inout_feat_dim, NodeAggregates, nullptr,
      nullptr, nullptr);
}

__global__ void HGTTriviallyEdgeParallelCompactAsOfNodeNodeMeanAggregation(
    int64_t* col_idxes, int64_t* etypes, int64_t* eids, float* EdgeMessages,
    float* EdgeAttnScores, int64_t num_nodes, int64_t num_edges,
    int64_t num_etypes, int64_t num_heads, int64_t inout_feat_dim,
    float* NodeAggregates, int64_t* ETypeUniqueIndexToNodeIndexMap,
    int64_t* etype_unique_node_offsets, int64_t* row_indices) {
  _HGTTriviallyEdgeParallelNodeMeanAggregation<
      int64_t, float, /* EdgeMessagesCompactAsOfNodeFlag = */ true,
      /* EdgeMessagesIndirectionOffsetInsteadOf2DArrayFlag = */ true, float*,
      /*BinarySearchToGetEtypeNodeOffsetFlag = */ true,
      /*CSRInsteadOfCOOFlag = */ false>(
      col_idxes, etypes, eids, EdgeMessages, EdgeAttnScores, num_nodes,
      num_edges, num_etypes, num_heads, inout_feat_dim, NodeAggregates,
      ETypeUniqueIndexToNodeIndexMap, etype_unique_node_offsets, row_indices);
}

// constexpr auto HGTTriviallyEdgeParallelVanillaNodeMeanAggregation =
// HGTTriviallyEdgeParallelNodeMeanAggregation<int, float, false, float *>;
// constexpr auto HGTTriviallyEdgeParallelCompactAsOfNodeNodeMeanAggregation =
// HGTTriviallyEdgeParallelNodeMeanAggregation<int, float, true, float **>;

// We need to use separate coo at this moment
// this kernel can be used for either sW in edge attention computation (vanilla
// + compactAsOfNode) or sW in edge message generation (vanilla). s STANDS FOR
// SOURCE NODE.
template <typename Idx, typename DType, int TILE_SZ_A, int TILE_SZ_B,
          int OUT_DIM, int NUM_HEADS, bool WORK_ASSIGNMENT_INDEX_FLAG,
          bool InputNodeFeaturesCompactOfNodeFlag,
          bool ETypeUniqueNodeIndexBinarySearchFlag>
__global__ void EdgeMessageGeneration(
    Idx* etype_edge_offsets, Idx* etype_block_offsets, Idx* row_idxes,
    Idx* col_idxes, Idx* etypes, Idx* eids, DType* weight, Idx num_nodes,
    Idx num_edges, Idx num_etypes, DType* NodeFeatures, DType* OutEdgeMessage,
    Idx* etype_unique_node_offsets, Idx* etype_unique_node_index_map) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_EDGES_PER_BLOCK = (TILE_SZ_B);

  int stride;
  int relation_idx;
  if constexpr (WORK_ASSIGNMENT_INDEX_FLAG) {
    assert(
        0 &&
        "WORK_ASSIGNMENT_INDEX_FLAG is not supported in EdgeMessageGeneration");
    // int beg_edge_entry_idx = beg_edge_entry_idxes_vect[blockIdx.x];
    // stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
    //             COARSE_SGEMM_EDGES_PER_BLOCK;
    // relation_idx = blockid_relation_id_vect[blockIdx.x];
  } else {
    relation_idx =
        binary_search<Idx, Idx*>(num_etypes, etype_block_offsets, blockIdx.x);
    stride =
        COARSE_SGEMM_EDGES_PER_BLOCK * (etype_block_offsets[relation_idx + 1] -
                                        etype_block_offsets[relation_idx]);
  }
  for (int edge_entry_idx = etype_edge_offsets[relation_idx];
       edge_entry_idx < etype_edge_offsets[relation_idx + 1];
       edge_entry_idx += stride) {
    if constexpr (InputNodeFeaturesCompactOfNodeFlag) {
      mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, true,
                      InputNodeFeaturesCompactOfNodeFlag,
                      ETypeUniqueNodeIndexBinarySearchFlag>::
          exec_function(
              OUT_DIM,
              etype_edge_offsets[relation_idx + 1] -
                  etype_edge_offsets[relation_idx],
              NODE_INPUT_DIM_PER_HEAD,
              &weight[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
                      NODE_INPUT_DIM_PER_HEAD],
              // assuming in_dim == out_dim
              &NodeFeatures[etype_unique_node_offsets[relation_idx] * OUT_DIM],
              OutEdgeMessage,
              &row_idxes[etype_edge_offsets[relation_idx]] /* source node
                           feature is what we want as one of the operand*/
              ,
              &etype_unique_node_index_map
                  [etype_unique_node_offsets[relation_idx]],
              etype_unique_node_offsets[relation_idx + 1] -
                  etype_unique_node_offsets[relation_idx],
              &eids[etype_edge_offsets[relation_idx]],
              edge_entry_idx - etype_edge_offsets[relation_idx]);
    } else {
      mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, true,
                      false, false>::
          exec_function(
              OUT_DIM, etype_edge_offsets[relation_idx + 1],
              NODE_INPUT_DIM_PER_HEAD,
              &weight[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
                      NODE_INPUT_DIM_PER_HEAD],
              // assuming in_dim == out_dim
              NodeFeatures, OutEdgeMessage,
              row_idxes /* source node feature is what we want as one of the
                           operand*/
              ,
              nullptr, -1, eids, edge_entry_idx);
    }
  }
}

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
constexpr auto HGTVanillaEdgeAttentionSecondStage =
    GeneralEdgeMessageMultiplyNodeFeature<
        float, /*ProductCompactAsOfNodeFlag = */ false,
        /*EidEnableFlag = */ true, float*>;

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
constexpr auto HGTCompactAsOfNodesEdgeAttentionSecondStage =
    GeneralEdgeMessageMultiplyNodeFeature<float, true, false, float**>;
