#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"
#include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"
#include "ThreadingGridsBlocksSchedules.h"

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {
namespace SeparateCOO {
namespace EdgeParallel {
void full_graph_edge_softmax_ops(
    at::Tensor &row_indices, at::Tensor &col_indices, at::Tensor &eids,
    at::Tensor &reltypes_ptr, at::Tensor &unnormalized_attn_score,
    at::Tensor &mu, at::Tensor &edgesoftmax_sum_per_node,
    at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  HET::TorchExport::HGT::FwProp::_full_graph_edge_softmax_ops<int64_t, float, 3,
                                                              true>(
      row_indices, col_indices, eids, reltypes_ptr, unnormalized_attn_score, mu,
      edgesoftmax_sum_per_node, mu_softmax_applied_unnormalized_attn_score,
      normalized_attn_score);
}
// adapted from HET::TorchExport::RGCN::FwProp::Layer1_SeparateCOO
void FullGraphFusedMessageCalcAndMeanAggregation(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_eids,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &node_feat_input, at::Tensor &weights, at::Tensor &edge_norm,
    /*at::Tensor& relation_pri, */ at::Tensor &node_feat_output) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  constexpr int WORK_BLOCK_SIZE = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X = true;
  constexpr bool COARSEN_FACTOR_2_FLAG_Y = true;
  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = weights.size(1);
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_dim = weights.size(3);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE), (int64_t)32768);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  // TODO: KWU: allow more dtype options in this file
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
  // TODO: KWU: allow more dtype options in this file
  HET_HGTMessageGenerationAndAccumulationFwProp<
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE,
      WORK_BLOCK_SIZE, WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          node_feat_input.data_ptr<float>(), weights.data_ptr<float>(),
          node_feat_output.data_ptr<float>(),
          edge_norm.data_ptr<float>(),  // relation_pri.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim, num_heads);
}

// TODO: KWU: use reg tiling here: test fuse attn score vs non-fused
// based on
// HGT::BckProp::SeparateCOO::EdgeParallel::FullGraphFusedMessageCalcAndMeanAggregation,
// i.e., wrapper function of
// HET_HGTMessageGenerationAndAccumulationDeltaWeightBckProp
void full_graph_hetero_attention_ops(
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_relptrs,
    at::Tensor &applied_klinear_node_features,
    at::Tensor &applied_qlinear_node_features, at::Tensor &attn_score_weight,
    at::Tensor &attn_score_inner_product, at::Tensor &unnormalized_attn_score) {
  // we need to implement a fused kernel based on W*t via RGNN relational_matmul
  // and RGNN inner_product
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // TODO: KWU: implement the switch to disable reg-tiling
  constexpr bool REG_TILING_FLAG = true;

  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = attn_score_weight.size(1);
  const int64_t num_input_dim = attn_score_weight.size(2);
  const int64_t num_output_dim = attn_score_weight.size(3);
  int64_t num_edges = separate_coo_eids.numel();

  // NB: configuration irrelavant to whether use reg tiled or not
  constexpr int WORK_BLOCK_SIZE_X = REG_TILING_FLAG ? 64 : 32;
  constexpr int WORK_BLOCK_SIZE_Y = REG_TILING_FLAG ? 16 : 32;
  constexpr int WORK_BLOCK_SIZE_K =
      REG_TILING_FLAG ? 16 : 32;  // TODO: KWU: change to 8

  int grid_dim_y = std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE_Y),
                            (int64_t)4096);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE_Y,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());

  // NB: KWU: choose reg tiled configurations by introducing ternary operators
  // NB: shmem-tiled specific configuration
  constexpr int THREADING_BLOCK_SIZE_X =
      REG_TILING_FLAG ? WORK_BLOCK_SIZE_X : WORK_BLOCK_SIZE_X / 2;
  constexpr int THREADING_BLOCK_SIZE_Y =
      REG_TILING_FLAG ? 1 : WORK_BLOCK_SIZE_Y / 2;
  // NB: my shmem sgemm matmul scheme
  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE_X),
                   grid_dim_y, num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);

  // NB: KWU: using reg tiled version by default
  HET_HGTFusedAttnScoreFwProp<THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y,
                              WORK_BLOCK_SIZE_X, WORK_BLOCK_SIZE_Y,
                              WORK_BLOCK_SIZE_K, int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          applied_klinear_node_features.data_ptr<float>(),
          applied_qlinear_node_features.data_ptr<float>(),
          attn_score_weight.data_ptr<float>(),
          attn_score_inner_product.data_ptr<float>(),
          unnormalized_attn_score.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim, num_heads);
}
}  // namespace EdgeParallel
}  // namespace SeparateCOO

}  // namespace FwProp
namespace BckProp {
namespace SeparateCOO {
namespace EdgeParallel {
void full_graph_hetero_attention_ops(
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &separate_coo_eids, at::Tensor &separate_coo_relptrs,
    at::Tensor &grad_attn_score_weight,
    at::Tensor &attn_score_weight_transposed,
    at::Tensor &applied_klinear_node_features,
    at::Tensor &applied_qlinear_node_features,
    at::Tensor &attn_score_inner_product, at::Tensor &grad_unnorm_attn_score,
    at::Tensor &grad_k, at::Tensor &grad_q) {
  // we need to implement a fused kernel based on back prop of RGNN
  // inner_product and back prop of W*t via RGNN relational_matmul

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int WORK_BLOCK_SIZE = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X = true;
  constexpr bool COARSEN_FACTOR_2_FLAG_Y = true;
  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;

  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = attn_score_weight_transposed.size(1);
  const int64_t num_fw_input_dim = attn_score_weight_transposed.size(3);
  const int64_t num_fw_output_dim = attn_score_weight_transposed.size(2);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE), (int64_t)32768);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());
  // NB: my shmem sgemm matmul scheme
  // NB: fw_input_dim is the actual delta_k feat dimension and therefore used to
  // determine nblks.x
  const dim3 nblks(ceil_div<>(num_fw_input_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, num_heads);
  // NB: delta_weight's feature is (num_heads, num_fw_input_dim,
  // num_fw_output_dim) and therefore nblks_outer_product.x is determined by
  // num_fw_output_dim
  const dim3 nblks_outer_product(
      ceil_div<>(num_fw_output_dim, (long)WORK_BLOCK_SIZE),
      ceil_div<>(num_fw_input_dim, (long)WORK_BLOCK_SIZE),
      num_heads * grid_dim_y);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);

  // delta_k = delta_inner_product*weight_transposed =
  // delta_attn_score*q*weight_transposed

  // delta_weight=delta_inner_product*k=delta_attn_score*q*k

  HET_HGTFusedAttnScoreDeltaKVectBckProp<
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE,
      WORK_BLOCK_SIZE, WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          applied_qlinear_node_features.data_ptr<float>(),
          attn_score_weight_transposed.data_ptr<float>(),
          grad_k.data_ptr<float>(), grad_unnorm_attn_score.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_fw_input_dim, num_fw_output_dim, num_heads);
  HET_HGTFusedAttnScoreDeltaWeightBckProp<
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE,
      WORK_BLOCK_SIZE, WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks_outer_product, nthrs, 0, stream>>>(
          applied_klinear_node_features.data_ptr<float>(),
          applied_qlinear_node_features.data_ptr<float>(),
          grad_attn_score_weight.data_ptr<float>(),
          grad_unnorm_attn_score.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_fw_input_dim, num_fw_output_dim, num_heads);

  // delta_q = delta_attn_score*inner_product
  BackwardToDeltaQData<int64_t, float> gdata{
      .num_heads = num_heads,
      .k_vect_dim_per_head = num_fw_output_dim,
      .eids = incsr_eids.data_ptr<int64_t>(),
      .grad_unnormalized_attn_score = grad_unnorm_attn_score.data_ptr<float>(),
      .k_inner_product = attn_score_inner_product.data_ptr<float>(),
      .grad_q_vectors = grad_q.data_ptr<float>()};

  // NB: Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
  // original seastar schedule to allow reduction within the warp, i.e., along
  // x-axis
  auto [nblks_type2, nthrs_type2] = get_type2_schedule(
      gdata.num_heads, gdata.k_vect_dim_per_head * gdata.num_heads,
      incsr_row_ptr.numel() - 1);
  // NB: message_out_dim is the total dim,  the number of elements for each head
  // is message_out_dim//num_heads

  HET__hgtQVectType2BackwardKernel<int64_t, float,
                                   CompactAsOfNodeKind::Disabled, true, true>
      <<<nblks_type2, nthrs_type2, 0, stream>>>(
          gdata, incsr_row_ptr.data_ptr<int64_t>(),
          incsr_col_indices.data_ptr<int64_t>(),
          incsr_reltypes.data_ptr<int64_t>(), incsr_row_ptr.numel() - 1, {},
          num_relations);
}

void FullGraphFusedMessageCalcAndMeanAggregation(
    at::Tensor &separate_coo_relptrs, at::Tensor &separate_coo_eids,
    at::Tensor &separate_coo_row_indices, at::Tensor &separate_coo_col_indices,
    at::Tensor &node_feat_input, at::Tensor &weights_transposed,
    at::Tensor &edge_norm,
    /*at::Tensor& relation_pri, */ at::Tensor &node_feat_output,
    at::Tensor &grad_node_feat_input, at::Tensor &grad_weights,
    at::Tensor &grad_edge_norm, at::Tensor &grad_node_feat_output) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int WORK_BLOCK_SIZE = 32;
  constexpr bool COARSEN_FACTOR_2_FLAG_X = true;
  constexpr bool COARSEN_FACTOR_2_FLAG_Y = true;
  constexpr int THREADING_BLOCK_SIZE_X =
      COARSEN_FACTOR_2_FLAG_X ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;
  constexpr int THREADING_BLOCK_SIZE_Y =
      COARSEN_FACTOR_2_FLAG_Y ? WORK_BLOCK_SIZE / 2 : WORK_BLOCK_SIZE;

  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = weights_transposed.size(1);
  const int64_t num_input_dim = weights_transposed.size(3);
  const int64_t num_output_dim = weights_transposed.size(2);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)WORK_BLOCK_SIZE), (int64_t)4096);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t *>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());
  // NB: my shmem sgemm matmul scheme
  // NB: nblks.x should be num_input_dim
  const dim3 nblks(ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE), grid_dim_y,
                   num_heads);
  const dim3 nblks_outer_product(
      ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
      ceil_div<>(num_input_dim, (long)WORK_BLOCK_SIZE), num_heads * grid_dim_y);
  const dim3 nthrs(THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y);
  HET_HGTMessageGenerationAndAccumulationDeltaNodeFeatInputBckProp<
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE,
      WORK_BLOCK_SIZE, WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks, nthrs, 0, stream>>>(
          grad_node_feat_output.data_ptr<float>(),
          weights_transposed.data_ptr<float>(),
          grad_node_feat_input.data_ptr<float>(),
          node_feat_input.data_ptr<float>(), edge_norm.data_ptr<float>(),
          grad_edge_norm.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_output_dim, num_input_dim, num_heads);
  HET_HGTMessageGenerationAndAccumulationDeltaWeightBckProp<
      THREADING_BLOCK_SIZE_X, THREADING_BLOCK_SIZE_Y, WORK_BLOCK_SIZE,
      WORK_BLOCK_SIZE, WORK_BLOCK_SIZE, int64_t, int64_t *>
      <<<nblks_outer_product, nthrs, 0, stream>>>(
          node_feat_input.data_ptr<float>(),
          grad_node_feat_output.data_ptr<float>(),
          grad_weights.data_ptr<float>(), edge_norm.data_ptr<float>(),
          separate_coo_row_indices.data_ptr<int64_t>(),
          separate_coo_col_indices.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim, num_heads);
}

}  // namespace EdgeParallel
}  // namespace SeparateCOO
}  // namespace BckProp

}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  m.def(
      "backward_hgt_full_graph_hetero_attention_ops_coo",
      HGT::BckProp::SeparateCOO::EdgeParallel::full_graph_hetero_attention_ops);
  m.def(
      "hgt_full_graph_hetero_attention_ops_coo",
      HGT::FwProp::SeparateCOO::EdgeParallel::full_graph_hetero_attention_ops);

  m.def("hgt_full_graph_edge_softmax_ops_separate_coo",
        HGT::FwProp::SeparateCOO::EdgeParallel::full_graph_edge_softmax_ops);

  m.def("hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo",
        HGT::FwProp::SeparateCOO::EdgeParallel::
            FullGraphFusedMessageCalcAndMeanAggregation);
  // clang-format off
  m.def(
      "backward_hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo",
      HGT::BckProp::SeparateCOO::EdgeParallel::
          FullGraphFusedMessageCalcAndMeanAggregation);
  // clang-format on
}
