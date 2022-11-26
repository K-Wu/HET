#pragma once
#include "DGLHackKernel/DGLHackKernel.h"
#include "DGLHackKernel/HGT/HGTBackwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTExperimental.h"
#include "DGLHackKernel/HGT/HGTForwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTPreprocessing.h"
#include "DGLHackKernel/OpPrototyping/HGTIntermediateData.h"
#include "DGLHackKernel/mysgemm/mysgemm_KernelsBlockConfigurations.h"
#include "EdgeAttention_4/EdgeAttentionCOO.h"

namespace HET {
namespace OpPrototyping {
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
  // num_blocks_for_same_relation_per_block_vect,
  // beg_node_entry_idxes_vect, blockid_relation_id_vect
  auto [num_blocks_for_same_relation_vect,
        num_blocks_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<true, false,
                                                      std::nullptr_t>(
          RTX_3090_GRIDSIZE,  // num_blocks
          hyper_params.num_relations, -1, nullptr, nullptr);

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  auto [num_blocks_for_same_relation_per_block_vect, blockid_relation_id_vect,
        beg_node_entry_idxes_vect] =
      get_schedule_by_relation_kernel_launch_per_block_metadata(
          num_blocks_for_same_relation_vect,
          num_blocks_for_all_prev_relation_vect, RTX_3090_GRIDSIZE,
          COARSE_SGEMM_NODES_PER_BLOCK);

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(num_blocks_for_all_prev_relation_vect
                [num_blocks_for_all_prev_relation_vect.size() - 1],
            2, 1);
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
          num_blocks_for_same_relation_per_block_vect.data()),
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
  // NODE_INPUT_DIM_PER_HEAD
  // float *relation_attention_matrices element num:
  // num_relations * NUM_HEADS * NODE_INPUT_DIM_PER_HEAD *
  // NODE_INPUT_DIM_PER_HEAD

  // preparing op kernel launch specific preprocessed metadata:
  // num_blocks_for_same_relation_per_block_vect,
  // beg_node_entry_idxes_vect, blockid_relation_id_vect

  auto [num_blocks_for_same_relation_vect,
        num_blocks_for_all_prev_relation_vect] =
      get_schedule_by_relation_kernel_launch_metadata<true, false,
                                                      std::nullptr_t>(
          RTX_3090_GRIDSIZE, hyper_params.num_relations, -1, nullptr, nullptr);

  // grid and thread configuration of the first stage
  //   block (0,0): (head0 (64 element), 16 nodes), (head1 (64 element), 16
  //   nodes); block(1,0): (head0 (64 element), 16 nodes), (head1 (64 element),
  //   16 nodes); ... block(BLOCKDIM_X-1,0): (head0 (64 element), 16 nodes),
  //   (head1 (64 element), 16 nodes); block (0,1): (head2 (64 element), 16
  //   nodes), (head3 (64 element), 16 nodes); block(1,1): (head2 (64 element),
  //   16 nodes), (head3 (64 element), 16 nodes); ... block(BLOCKDIM_X-1,1):
  //   (head2 (64 element), 16 nodes), (head3 (64 element), 16 nodes);

  auto [num_blocks_for_same_relation_per_block_vect, blockid_relation_id_vect,
        beg_node_entry_idxes_vect] =
      get_schedule_by_relation_kernel_launch_per_block_metadata(
          num_blocks_for_same_relation_vect,
          num_blocks_for_all_prev_relation_vect, RTX_3090_GRIDSIZE,
          COARSE_SGEMM_NODES_PER_BLOCK);

  dim3 block(COARSE_SGEMM_BLOCKSIZE, 1, 1);
  dim3 grid(num_blocks_for_all_prev_relation_vect
                [num_blocks_for_all_prev_relation_vect.size() - 1],
            2, 1);
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
          num_blocks_for_same_relation_per_block_vect.data()),
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
      thrust::raw_pointer_cast(concatenated_coo_matrix_values.data()), nullptr,
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

template </*int XPU, */ typename Idx, typename DType>
void HGTBackPropGradientSMAFusion(
    // GraphRef graph,
    MyHeteroIntegratedCSR<Idx, thrust::device_allocator<Idx>> csr,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>&
        grad_sm_first_stage,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_a,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& grad_t_neighbour,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& message,
    MySimpleNDArray<DType, thrust::device_allocator<DType>>& sigmas) {
  // LOG(INFO) << "Calling implementation of rgn layer 1 forward";
  // assert(csr.IsSortedByEdgeType_CPU());
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(false);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];
  auto range_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.row_ptr.data()));
  auto ids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.col_idx.data()));
  // auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids);
  auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(csr.eids.data()));
  auto typeids_data =
      static_cast<Idx*>(thrust::raw_pointer_cast(csr.rel_type.data()));
  auto grad_sm_first_stage_data = grad_sm_first_stage.Ptr();
  auto grad_a_data = grad_a.Ptr();
  auto grad_t_neighbour_data = grad_t_neighbour.Ptr();
  auto message_data = message.Ptr();
  auto sigmas_data = sigmas.Ptr();

  // print_dims(hidden);
  // print_dims(weight);
  // print_dims(norm);
  // print_dims(ret);
  // Idx num_nodes = ranges->shape[0] - 1;
  // Idx num_edges = eids->shape[0];
  Idx num_nodes = csr.num_rows;
  Idx num_edges = csr.col_idx.size();
  Idx num_heads = grad_sm_first_stage.shape[2];
  Idx feat_dim_per_head = grad_sm_first_stage.shape[3];
  Idx n_rel_types = grad_sm_first_stage.shape[1];
  int nblks = num_nodes;
  int nthrs = num_heads * feat_dim_per_head;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  HGTBackwardGradientSmFirstPartImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_sm_first_stage_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "HGTBackwardGradientSmFirstPartImpl time: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << " ms" << std::endl;

  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1_kernel2 =
      std::chrono::high_resolution_clock::now();
  HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_a_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2_kernel2 =
      std::chrono::high_resolution_clock::now();
  std::cout << "HGTBackwardGradientAImpl time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   t2_kernel2 - t1_kernel2)
                   .count()
            << " ms" << std::endl;

  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t1_kernel3 =
      std::chrono::high_resolution_clock::now();

  HGTBackwardFusedGradientSmFirstPartGradientAImpl<Idx, DType>
      <<<nblks, nthrs>>>(range_data, ids_data, eids_data, typeids_data,
                         grad_a_data, grad_sm_first_stage_data,
                         grad_t_neighbour_data, message_data, sigmas_data,
                         num_nodes, num_heads, feat_dim_per_head, n_rel_types);
  cuda_err_chk(cudaPeekAtLastError());
  cuda_err_chk(cudaDeviceSynchronize());
  std::chrono::high_resolution_clock::time_point t2_kernel3 =
      std::chrono::high_resolution_clock::now();
  std::cout << "HGTBackwardFusedGradientSmFirstPartGradientAImpl time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   t2_kernel3 - t1_kernel3)
                   .count()
            << " ms" << std::endl;
}

}  // namespace OpPrototyping
}  // namespace HET