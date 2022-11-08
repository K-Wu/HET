#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGT/HGTPreprocessing.h"
#include "EdgeAttention_4/mysgemm_functor.cu.h"

// extract this kernel with mysgemm_ into template specialization
// template <int NODE_INPUT_DIM_PER_HEAD/*derived from OUT_DIM and NUM_HEADS*/,
// NUM_HEADS, OUT_DIM, COARSE_SGEMM_NODES_PER_BLOCK>
template <int TILE_SZ_A, int TILE_SZ_B, int OUT_DIM, int NUM_HEADS>
__global__ void _global_EdgeMessageConcatenatedCOOKernel(
    float **__restrict__ intermediate_node_vect, int nnz,
    int *__restrict__ matCols, int *__restrict__ matRelation,
    float *__restrict__ node_input_data,
    float *__restrict__ relation_message_matrices,
    int **__restrict__ dest_node_to_unique_index_per_relation,
    int *__restrict__ sizes_unique_index_to_dest_node_per_relation,
    int num_relations,
    int *__restrict__ num_blocks_xdim_for_same_relation_per_block_vect,
    int *__restrict__ beg_node_entry_idxes_vect,
    int *__restrict__ blockid_relation_id_vect) {
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
