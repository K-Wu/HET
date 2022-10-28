#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGTPreprocessing.h"
#include "EdgeAttention_4/EdgeAttentionCOO.h"
#include "EdgeAttention_4/mysgemm_functor.cu.h"
#include "utils/cuda_helper_device_functions.cu.h"

// only requires column indices and therefore both COO and CSR could leverage
// this kernel.
template <typename Idx, typename DType>
__global__ void HGTTriviallyEdgeParallelNodeMeanAggregation(
    Idx* col_idxes, Idx* etypes, Idx* eids, DType* EdgeMessages,
    DType* EdgeAttnScores, Idx num_nodes, Idx num_edges, Idx num_etypes,
    Idx num_heads, Idx inout_feat_dim, DType* NodeAggregates) {
  // each warp deals with one edge
  Idx edge_idx =
      (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  // warpSize = 32
  static_assert(inout_feat_dim % warpSize == 0,
                "inout_feat_dim must be multiple of warpSize");  // 32
  for (; edge_idx < num_edges; edge_idx += gridDim.x * blockDim.x / warpSize) {
    Idx col_idx = col_idxes[edge_idx];
    Idx etype = etypes[edge_idx];
    Idx eid = eids[edge_idx];
    DType* EdgeMessage = &EdgeMessages[eid * inout_feat_dim];
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

// We need to use separate coo at this moment
// this kernel can be used for either sW in edge attention computation or sW in
// edge message generation. s STANDS FOR SOURCE NODE.
template <typename Idx, typename DType, int TILE_SZ_A, int TILE_SZ_B,
          int OUT_DIM, int NUM_HEADS>
__global__ void EdgeMessageComputation(Idx* etype_edge_offsets, Idx* row_idxes,
                                       Idx* col_idxes, Idx* etypes, Idx* eids,
                                       DType* weight, Idx num_nodes,
                                       Idx num_edges, Idx num_etypes,
                                       DType* NodeFeatures,
                                       DType* OutEdgeMessage) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_EDGES_PER_BLOCK = (TILE_SZ_B);
  int beg_edge_entry_idx = beg_edge_entry_idxes_vect[blockIdx.x];
  int stride = num_blocks_xdim_for_same_relation_per_block_vect[blockIdx.x] *
               COARSE_SGEMM_EDGES_PER_BLOCK;
  int relation_idx = blockid_relation_id_vect[blockIdx.x];

  for (int edge_entry_idx = beg_edge_entry_idx;
       edge_entry_idx <
       etype_edge_offsets[relation_idx + 1] - etype_edge_offsets[relation_idx];
       edge_entry_idx += stride) {
    mysgemm_functor<TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS, true, true>::
        exec_function(
            OUT_DIM,
            etype_edge_offsets[relation_idx + 1] -
                etype_edge_offsets[relation_idx],
            NODE_INPUT_DIM_PER_HEAD,
            &weight[relation_idx * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
                    NODE_INPUT_DIM_PER_HEAD],
            NodeFeatures, OutEdgeMessage,
            row_idxes /* source node feature is what we want as one of the
                         operand*/
            ,
            eids, edge_entry_idx);
  }
}

// This is to calculate the product of (sW) and t where (sW) is stored per edge
// and t is stored per node.
constexpr auto HGTVanillaEdgeAttentionSecondStage =
    EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<
        float, false, true, float*>;