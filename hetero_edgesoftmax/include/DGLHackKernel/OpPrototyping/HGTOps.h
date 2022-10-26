#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGTExperimental.h"
#include "DGLHackKernel/HGTLayersBackwardKernels.cu.h"
#include "DGLHackKernel/HGTLayersKernels.cu.h"
#include "DGLHackKernel/HGTPreprocessing.h"
#include "DGLHackKernel/OpPrototyping/HGTIntermediateData.h"
#include "EdgeAttention_4/EdgeAttentionCOO.h"

struct HGTLayerWeights {
  MySimpleNDArray<float, thrust::device_allocator<float>> KLinearWeights;
  MySimpleNDArray<float, thrust::device_allocator<float>> KLinearBias;
  MySimpleNDArray<float, thrust::device_allocator<float>> QLinearWeights;
  MySimpleNDArray<float, thrust::device_allocator<float>> QLinearBias;
  MySimpleNDArray<float, thrust::device_allocator<float>> VLinearWeights;
  MySimpleNDArray<float, thrust::device_allocator<float>> VLinearBias;
  MySimpleNDArray<float, thrust::device_allocator<float>> ALinearWeights;
  MySimpleNDArray<float, thrust::device_allocator<float>> ALinearBias;
  MySimpleNDArray<float, thrust::device_allocator<float>>
      relation_attention_matrices;
  MySimpleNDArray<float, thrust::device_allocator<float>>
      relation_message_matrices;
  HGTLayerWeights(
      MySimpleNDArray<float, thrust::device_allocator<float>>& KLinearWeights,
      MySimpleNDArray<float, thrust::device_allocator<float>>& KLinearBias,
      MySimpleNDArray<float, thrust::device_allocator<float>>& QLinearWeights,
      MySimpleNDArray<float, thrust::device_allocator<float>>& QLinearBias,
      MySimpleNDArray<float, thrust::device_allocator<float>>& VLinearWeights,
      MySimpleNDArray<float, thrust::device_allocator<float>>& VLinearBias,
      MySimpleNDArray<float, thrust::device_allocator<float>>& ALinearWeights,
      MySimpleNDArray<float, thrust::device_allocator<float>>& ALinearBias,
      MySimpleNDArray<float, thrust::device_allocator<float>>&
          relation_attention_matrices,
      MySimpleNDArray<float, thrust::device_allocator<float>>&
          relation_message_matrices)
      : KLinearWeights(KLinearWeights),
        KLinearBias(KLinearBias),
        QLinearWeights(QLinearWeights),
        QLinearBias(QLinearBias),
        VLinearWeights(VLinearWeights),
        VLinearBias(VLinearBias),
        ALinearWeights(ALinearWeights),
        ALinearBias(ALinearBias),
        relation_attention_matrices(relation_attention_matrices),
        relation_message_matrices(relation_message_matrices) {}
};

std::shared_ptr<HGTLayerWeights> InitializeHGTLayerWeights(
    HGTLayerHyperParams hyper_params) {
  // TODO: implement move constructor for either HGTLayerWeights or
  // MySimpleNDArray to reduce memory footprint during initialization.
  MySimpleNDArray<float, thrust::device_allocator<float>> KLinearWeights =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.klinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> KLinearBias =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.klinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> QLinearWeights =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.qlinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> QLinearBias =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.qlinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> VLinearWeights =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.input_dim, hyper_params.vlinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> VLinearBias =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_node_types, hyper_params.num_heads,
           hyper_params.vlinear_out_dim});
  // float *node_input_data; element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD float *relation_attention_matrices; element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD
  MySimpleNDArray<float, thrust::device_allocator<float>>
      relation_attention_matrices =
          MySimpleNDArray<float, thrust::device_allocator<float>>(
              {hyper_params.num_relations, hyper_params.num_heads,
               hyper_params.klinear_out_dim, hyper_params.qlinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>>
      relation_message_matrices =
          MySimpleNDArray<float, thrust::device_allocator<float>>(
              {hyper_params.num_relations, hyper_params.num_heads,
               hyper_params.vlinear_out_dim, hyper_params.message_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> ALinearWeights =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_heads, hyper_params.message_dim,
           hyper_params.alinear_out_dim});
  MySimpleNDArray<float, thrust::device_allocator<float>> ALinearBias =
      MySimpleNDArray<float, thrust::device_allocator<float>>(
          {hyper_params.num_heads, hyper_params.alinear_out_dim});
  KLinearWeights.FillInRandomData();
  KLinearBias.FillInRandomData();
  QLinearWeights.FillInRandomData();
  QLinearBias.FillInRandomData();
  VLinearWeights.FillInRandomData();
  VLinearBias.FillInRandomData();
  ALinearWeights.FillInRandomData();
  ALinearBias.FillInRandomData();
  relation_attention_matrices.FillInRandomData();
  relation_message_matrices.FillInRandomData();
  return std::make_shared<HGTLayerWeights>(
      KLinearWeights, KLinearBias, QLinearWeights, QLinearBias, VLinearWeights,
      VLinearBias, ALinearWeights, ALinearBias, relation_attention_matrices,
      relation_message_matrices);
}

// NB: In this implementation, message generation is done for each edge. This
// means involving repetitive computation during the GEMM phase.
void VanillaEdgeMessageConcatenatedCOOKernel(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data,
    std::shared_ptr<HGTLayerExecPreprocessedData> preprocessed_data,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values) {
  assert(0 && "not implemented");
}

// NB: In this implementation, message generation is done for each (source node,
// relationship this node is involved) where each (source node, relationship
// this node is involved) is mapped to a unique (relationship id, unique node
// index) and referred to in the next stage. Notice getting this unique index
// mapping is O(|R||V|) complexity and stays the same throughout the whole
// execution. We can do this mapping in the first step and reuse it thereafter.
// In this case, the it is dense operation first with scatter operation
// implicitly done by succeeding operations.
// TODO: an alternative implementation is message generation for each edge where
// there might be redundant computation of (source node, relationship this node
// is involved) pairs. In this case, only the relationship type and source node
// index for each edge is needed. This is explicit scatter operation done first
// and then dense operation.
template <int TILE_SZ_A /*128*/, int TILE_SZ_B /*8*/, int OUT_DIM /*256*/,
          int NUM_HEADS /*4*/>
void CompressedEdgeMessageConcatenatedCOOKernel(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data,
    std::shared_ptr<HGTLayerExecPreprocessedData> preprocessed_data,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  // float *node_input_data element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD float *relation_attention_matrices element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD

  // preparing op kernel launch specific preprocessed metadata:
  // num_blocks_xdim_for_same_relation_per_block_vect,
  // beg_node_entry_idxes_vect, blockid_relation_id_vect
  thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;
  std::vector<int> num_blocks_xdim_for_same_relation_vect;
  std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
  num_blocks_xdim_for_all_prev_relation_vect.push_back(0);

  // for ease of programming equally partition the workload to different blocks
  // at this moment.
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    int num_blocks_xdim_for_this_and_prev_relation =
        (idx_relationship + 1 + 0.0) / (hyper_params.num_relations + 0.0) *
        RTX_3090_GRIDSIZE;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(
        num_blocks_xdim_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    num_blocks_xdim_for_same_relation_vect.push_back(
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_xdim_for_all_prev_relation_vect.erase(
      num_blocks_xdim_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation <
            num_blocks_xdim_for_all_prev_relation_vect.size() - 1 &&
        idx_block >=
            num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK ==
             num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
    num_blocks_xdim_for_same_relation_per_block_vect.push_back(
        num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 2, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  // v*W
  EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<
      TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS><<<grid, block>>>(
      thrust::raw_pointer_cast(
          (intermediate_data->get_intermediate_node_vect_d()).data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      (intermediate_data->VLinearOutput).Ptr(),
      (weights->relation_message_matrices).Ptr(),
      thrust::raw_pointer_cast(
          (preprocessed_data->get_dest_node_to_unique_index_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          (preprocessed_data
               ->get_unique_indices_to_column_indices_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          (preprocessed_data->num_unique_indices_to_column_indices_per_relation)
              .data()),
      hyper_params.num_relations,
      thrust::raw_pointer_cast(
          num_blocks_xdim_for_same_relation_per_block_vect.data()),
      thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()),
      thrust::raw_pointer_cast(blockid_relation_id_vect.data()));

  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeAttentionConcatenatedCOO_128_8 Kernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  return;
}

template <int TILE_SZ_A /*128*/, int TILE_SZ_B /*8*/, int OUT_DIM /*256*/,
          int NUM_HEADS /*4*/>
void EdgeAttentionConcatenatedCOOKernel(
    HGTLayerHyperParams hyper_params, std::shared_ptr<HGTLayerWeights> weights,
    std::shared_ptr<HGTLayerIntermediateData> intermediate_data,
    std::shared_ptr<HGTLayerExecPreprocessedData> preprocessed_data,
    cusp::coo_matrix<int, int, cusp::device_memory>::row_indices_array_type
        concatenated_coo_matrix_row_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::column_indices_array_type
        concatenated_coo_matrix_column_indices,
    std::vector<cusp::coo_matrix<
        int, int, cusp::device_memory>::column_indices_array_type>
        coo_matrices_column_indices,
    cusp::coo_matrix<int, int, cusp::device_memory>::values_array_type
        concatenated_coo_matrix_values) {
  constexpr int NODE_INPUT_DIM_PER_HEAD = (OUT_DIM / NUM_HEADS);
  constexpr int COARSE_SGEMM_BLOCKSIZE = (TILE_SZ_A);
  constexpr int COARSE_SGEMM_NODES_PER_BLOCK = (TILE_SZ_B);

  // float *node_input_data element num: num_nodes * NUM_HEADS *
  // NODE_INPUT_DIM_PER_HEAD float *relation_attention_matrices element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD

  // preparing op kernel launch specific preprocessed metadata:
  // num_blocks_xdim_for_same_relation_per_block_vect,
  // beg_node_entry_idxes_vect, blockid_relation_id_vect
  thrust::device_vector<int> num_blocks_xdim_for_same_relation_per_block_vect;
  thrust::device_vector<int> blockid_relation_id_vect;
  thrust::device_vector<int> beg_node_entry_idxes_vect;
  std::vector<int> num_blocks_xdim_for_same_relation_vect;
  std::vector<int> num_blocks_xdim_for_all_prev_relation_vect;
  num_blocks_xdim_for_all_prev_relation_vect.push_back(0);

  // for ease of programming equally partition the workload to different blocks
  // at this moment.
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    int num_blocks_xdim_for_this_and_prev_relation =
        (idx_relationship + 1 + 0.0) / (hyper_params.num_relations + 0.0) *
        RTX_3090_GRIDSIZE;
    num_blocks_xdim_for_all_prev_relation_vect.push_back(
        num_blocks_xdim_for_this_and_prev_relation);
  }
  for (int idx_relationship = 0; idx_relationship < hyper_params.num_relations;
       idx_relationship++) {
    num_blocks_xdim_for_same_relation_vect.push_back(
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship + 1] -
        num_blocks_xdim_for_all_prev_relation_vect[idx_relationship]);
  }
  num_blocks_xdim_for_all_prev_relation_vect.erase(
      num_blocks_xdim_for_all_prev_relation_vect.begin());
  int idx_curr_relation = 0;
  int curr_beg_node_entry_idx = 0;

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  for (int idx_block = 0; idx_block < RTX_3090_GRIDSIZE; idx_block++) {
    if (idx_curr_relation <
            num_blocks_xdim_for_all_prev_relation_vect.size() - 1 &&
        idx_block >=
            num_blocks_xdim_for_all_prev_relation_vect[idx_curr_relation]) {
      assert(curr_beg_node_entry_idx / COARSE_SGEMM_NODES_PER_BLOCK ==
             num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
      idx_curr_relation++;
      curr_beg_node_entry_idx = 0;
    }
    blockid_relation_id_vect.push_back(idx_curr_relation);
    beg_node_entry_idxes_vect.push_back(curr_beg_node_entry_idx);
    curr_beg_node_entry_idx += COARSE_SGEMM_NODES_PER_BLOCK;
    num_blocks_xdim_for_same_relation_per_block_vect.push_back(
        num_blocks_xdim_for_same_relation_vect[idx_curr_relation]);
  }

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(RTX_3090_GRIDSIZE, 2, 1);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  // k*W
  EdgeAttentionConcatenatedFirstStageWeightMulDestCOOKernel<
      TILE_SZ_A, TILE_SZ_B, OUT_DIM, NUM_HEADS><<<grid, block>>>(
      thrust::raw_pointer_cast(
          (intermediate_data->get_intermediate_node_vect_d()).data()),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      (intermediate_data->KLinearOutput).Ptr(),
      (weights->relation_attention_matrices).Ptr(),
      thrust::raw_pointer_cast(
          (preprocessed_data->get_dest_node_to_unique_index_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          (preprocessed_data
               ->get_unique_indices_to_column_indices_per_relation_d())
              .data()),
      thrust::raw_pointer_cast(
          (preprocessed_data->num_unique_indices_to_column_indices_per_relation)
              .data()),
      hyper_params.num_relations,
      thrust::raw_pointer_cast(
          num_blocks_xdim_for_same_relation_per_block_vect.data()),
      thrust::raw_pointer_cast(beg_node_entry_idxes_vect.data()),
      thrust::raw_pointer_cast(blockid_relation_id_vect.data()));

  dim3 block2(RTX_3090_BLOCKSIZE, 1, 1);
  dim3 grid2(RTX_3090_GRIDSIZE, 1, 1);
  // (kW)*q
  EdgeAttentionConcatenatedSecondStageSrcInnerProductDestIntemediateCOOKernel<<<
      grid2, block2>>>(
      (intermediate_data->EdgeAttention).Ptr(),
      concatenated_coo_matrix_column_indices.size(),
      thrust::raw_pointer_cast(concatenated_coo_matrix_column_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_row_indices.data()),
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()),
      (intermediate_data->QLinearOutput).Ptr(),
      thrust::raw_pointer_cast(
          (intermediate_data->get_intermediate_node_vect_d()).data()),
      thrust::raw_pointer_cast(
          (preprocessed_data->get_dest_node_to_unique_index_per_relation_d())
              .data()));

  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "GPU doGPUEdgeAttentionConcatenatedCOO_128_8 Kernel time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " us" << std::endl;
  return;
}
