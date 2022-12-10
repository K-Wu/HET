#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {
namespace SeparateCOO {
namespace EdgeParallel {
// adapted from HET::TorchExport::RGCN::FwProp::Layer1_SeparateCOO
void FullGraphFusedMessageCalcAndMeanAggregation(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_row_idx, at::Tensor& separate_coo_col_idx,
    at::Tensor& node_feat_input, at::Tensor& weights, at::Tensor& edge_norm,
    /*at::Tensor& relation_pri, */ at::Tensor& node_feat_output) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int THREADING_BLOCK_SIZE = 16;
  constexpr bool COARSEN_FACTOR_2_FLAG = true;
  constexpr int WORK_BLOCK_SIZE =
      COARSEN_FACTOR_2_FLAG ? (THREADING_BLOCK_SIZE * 2) : THREADING_BLOCK_SIZE;
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
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
          grid_dim_y, num_relations, WORK_BLOCK_SIZE,
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>(),
          separate_coo_relptrs_cpu_contiguous.data_ptr<int64_t>() +
              num_relations + 1);
  grid_dim_y = num_blocks_assignment_for_all_prev_relation_vect.back();

  thrust::device_vector<int> dev_num_blocks_assignment_for_same_relation_vect(
      num_blocks_assignment_for_same_relation_vect.begin(),
      num_blocks_assignment_for_same_relation_vect.end());
  thrust::device_vector<int>
      dev_num_blocks_assignment_for_all_prev_relation_vect(
          num_blocks_assignment_for_all_prev_relation_vect.begin(),
          num_blocks_assignment_for_all_prev_relation_vect.end());

  const dim3 nblks(ceil_div<>(num_output_dim, (long)WORK_BLOCK_SIZE),
                   grid_dim_y, num_heads);
  const dim3 nthrs(THREADING_BLOCK_SIZE, THREADING_BLOCK_SIZE);
  HET_HGTMessageGenerationAndAccumulationFwProp<
      COARSEN_FACTOR_2_FLAG, THREADING_BLOCK_SIZE, int64_t, int64_t*>
      <<<nblks, nthrs, 0, stream>>>(
          node_feat_input.data_ptr<float>(), weights.data_ptr<float>(),
          node_feat_output.data_ptr<float>(),
          edge_norm.data_ptr<float>(),  // relation_pri.data_ptr<float>(),
          separate_coo_row_idx.data_ptr<int64_t>(),
          separate_coo_col_idx.data_ptr<int64_t>(),
          separate_coo_eids.data_ptr<int64_t>(),
          separate_coo_relptrs.data_ptr<int64_t>(),
          thrust::raw_pointer_cast(
              dev_num_blocks_assignment_for_all_prev_relation_vect.data()),
          num_relations, num_input_dim, num_output_dim, num_heads);
}
}  // namespace EdgeParallel
}  // namespace SeparateCOO
}  // namespace FwProp
namespace IntegratedCSR {
namespace EdgeParallel {
void full_graph_message_mean_aggregation() {
  // We may use HET_HGTTriviallyEdgeParallelCompactAsOfNodeNodeMeanAggregation
  // in hetero_edgesoftmax/include/DGLHackKernel/HGT/HGTForwardKernels.cu.h
  assert(0 && "Not implemented yet");
}
void full_graph_edge_softmax_ops() { assert(0 && "Not implemented yet"); }
}  // namespace EdgeParallel
}  // namespace IntegratedCSR
namespace BckProp {
namespace SeparateCOO {
namespace EdgeParallel {
void FullGraphFusedMessageCalcAndMeanAggregation(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_row_idx, at::Tensor& separate_coo_col_idx,
    at::Tensor& node_feat_input, at::Tensor& weights_transposed,
    at::Tensor& edge_norm,
    /*at::Tensor& relation_pri, */ at::Tensor& node_feat_output,
    at::Tensor& grad_node_feat_input, at::Tensor& grad_weights,
    at::Tensor& grad_edge_norm, at::Tensor& grad_node_feat_output, ) {
  assert(0 && "Not implemented yet");
}

}  // namespace EdgeParallel
}  // namespace SeparateCOO
}  // namespace BckProp
namespace IntegratedCSR {
namespace EdgeParallel {
void full_graph_message_mean_aggregation() {
  assert(0 && "Not implemented yet");
}
void full_graph_edge_softmax_ops() { assert(0 && "Not implemented yet"); }
}  // namespace EdgeParallel
}  // namespace IntegratedCSR
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET
