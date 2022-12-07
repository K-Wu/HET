#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/HGT/HGTBackwardExperimental.cu.h"
#include "DGLHackKernel/HGT/HGTBackwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTForwardExperimental.cu.h"
#include "DGLHackKernel/HGT/HGTForwardKernels.cu.h"
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

// TODO: add torch tensor version of HGTForwardImpl from
// [[hetero_edgesoftmax/include/DGLHackKernel/OpPrototyping/HGTProtoOps.h]]

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {

// adapted from HET::TorchExport::RGCN::FwProp::Layer1_SeparateCOO
void FullGraphFusedMessageCalcAndMeanAggregationSeparateCOO(
    at::Tensor& separate_coo_relptrs, at::Tensor& separate_coo_eids,
    at::Tensor& separate_coo_row_idx, at::Tensor& separate_coo_col_idx,
    at::Tensor& node_feat_input, at::Tensor& weights, at::Tensor& edge_norm,
    /*at::Tensor& relation_pri, */ at::Tensor& node_feat_output) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int BLOCK_SIZE = 16;
  const int64_t num_relations = (separate_coo_relptrs.numel() - 1);
  const int64_t num_heads = weights.size(1);
  const int64_t num_input_dim = weights.size(2);
  const int64_t num_output_dim = weights.size(3);
  int64_t num_edges = separate_coo_eids.numel();
  int grid_dim_y =
      std::min(ceil_div<>(num_edges, (int64_t)BLOCK_SIZE), (int64_t)32768);
  at::Tensor separate_coo_relptrs_cpu_contiguous =
      separate_coo_relptrs.cpu().contiguous();
  std::vector<int> num_blocks_assignment_for_same_relation_vect,
      num_blocks_assignment_for_all_prev_relation_vect;
  std::tie(num_blocks_assignment_for_same_relation_vect,
           num_blocks_assignment_for_all_prev_relation_vect) =
      get_schedule_by_relation_kernel_launch_metadata<false, false, int64_t*>(
          grid_dim_y, num_relations, BLOCK_SIZE,
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

  const dim3 nblks(ceil_div<>(num_output_dim, (long)BLOCK_SIZE), grid_dim_y,
                   num_heads);
  const dim3 nthrs(BLOCK_SIZE, BLOCK_SIZE);
  HGTMessageGenerationAndAccumulationFwProp<BLOCK_SIZE, int64_t, int64_t*>
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

namespace IntegratedCSR {
template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor& incsr_rowptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_reltypes, at::Tensor& incsr_eids,
    at::Tensor& edge_messages, at::Tensor& edge_attn_score,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& mu, at::Tensor& ret) {
  // using _hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum based on
  // _gatSumProdZipDivKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]] NB:
  // based on (vertex-centric) _gatSumProdZipDivKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]] or
  // (edge-centric) _gatSumProdZipDivKernel_edge_parallel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // configure parameter struct
  HgtDstOutData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads = edge_attn_score.size(edge_attn_score.ndimension() - 1);
  Idx num_relations = mu.numel() / num_heads;
  gdata.num_heads = num_heads;
  gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
  assert(gdata.num_heads ==
             edge_messages.size(edge_messages.ndimension() - 2) &&
         "assuming edge_messages[-2] to be num_heads but failed");
  gdata.message_out_dim =
      edge_messages.size(edge_messages.ndimension() - 1) * gdata.num_heads;
  gdata.eids = incsr_eids.data_ptr<Idx>();
  gdata.message = edge_messages.data_ptr<DType>();
  gdata.ret = ret.data_ptr<DType>();
  if constexpr (UseMuAppliedAttnScoreSwitch == 0) {
    gdata.unnormalized_attn_score = edge_attn_score.data_ptr<DType>();
    gdata.mu = mu.data_ptr<DType>();
  } else if constexpr (UseMuAppliedAttnScoreSwitch == 1) {
    gdata.mu_softmax_applied_unnormalized_attn_score =
        edge_attn_score.data_ptr<DType>();
  } else if constexpr (UseMuAppliedAttnScoreSwitch == 2) {
    gdata.normalized_attn_score = edge_attn_score.data_ptr<DType>();
  } else {
    assert(0 &&
           "invalid UseMuAppliedAttnScoreSwitch in "
           "HET::TorchExport::HGT::FwProp::IntegratedCSR::_full_graph_message_"
           "mean_aggregation");
  }

  Idx incsr_num_rows = incsr_rowptr.numel() - 1;
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::FwProp::IntegratedCSR::_full_graph_"
  //                "message_mean_aggregation"
  //             << std::endl;
  // }

  // Configure kernel launch parameters.
  int nthrs_x = SeastarFindNumThreads(num_heads, 64);
  int nthrs_y = SeastarFindNumThreads(
      gdata.message_out_dim,
      MAX_NTHRS /
          nthrs_x);  // NB: message_out_dim is the total dim, and the number of
                     // elements for each head is message_out_dim//num_heads
  int nblks_x = 1;
  int nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  printf("nblks_x=%d, nblks_y=%d, nthrs_x=%d, nthrs_y=%d\n", nblks_x, nblks_y,
         nthrs_x, nthrs_y);
  _hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum<
      Idx, DType, false, true, false, false, UseMuAppliedAttnScoreSwitch>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, incsr_rowptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
          incsr_reltypes.data_ptr<Idx>(), incsr_num_rows,
          nullptr /*no need when !CompactAsOfNodeFlag*/,
          nullptr /*no need when !CompactAsOfNodeFlag*/, num_relations);
}

void full_graph_message_mean_aggregation(
    at::Tensor& incsr_rowptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_reltypes, at::Tensor& incsr_eids,
    at::Tensor& edge_messages, at::Tensor& edge_attn_score,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& mu, at::Tensor& ret) {
  _full_graph_message_mean_aggregation<int64_t, float, 1>(
      incsr_rowptr, incsr_col_idx, incsr_reltypes, incsr_eids, edge_messages,
      edge_attn_score, edgesoftmax_sum_per_node, mu, ret);
}

void full_graph_hetero_attention_ops(at::Tensor& row_ptr, at::Tensor& col_idx,
                                     at::Tensor& eids, at::Tensor& reltypes,
                                     at::Tensor& weight,
                                     at::Tensor& applied_klinear_node_features,
                                     at::Tensor& applied_qlinear_node_features,
                                     at::Tensor& out) {
  // we need to implement a fused kernel based on W*t via RGNN relational_matmul
  // and RGNN inner_product

  assert(0 && "the performant implementation not done yet");
}

// this function 1) (accumulation stage) calculates edge softmax sum at each
// destination node, and 2) (scatter stage) normalize the attention score by the
// sum at each edge
template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch,
          bool EdgeParallelFlag>
void _full_graph_edge_softmax_ops(
    at::Tensor& incsr_rowptr_or_indices, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes_or_relptr,
    at::Tensor& unnormalized_attn_score, at::Tensor& mu,
    at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score) {
  // using _hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
  // There is an existing implementation with tricky API in
  // hetero_edgesoftmax/include/EdgeSoftmax_1/EdgeSoftmaxCSR.h
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // Configure kernel parameters structure
  HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads =
      unnormalized_attn_score.size(unnormalized_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::FwProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / num_heads;

  gdata.num_heads = num_heads;
  gdata.eids = incsr_eids.data_ptr<Idx>();
  gdata.mu = mu.data_ptr<DType>();
  gdata.unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>();
  gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
  if constexpr (OutputMuAppliedAttnScoreSwitch == 1) {
    gdata.mu_softmax_applied_unnormalized_attn_score =
        mu_softmax_applied_unnormalized_attn_score.data_ptr<DType>();
  } else if constexpr (OutputMuAppliedAttnScoreSwitch == 2) {
    gdata.normalized_attn_score = normalized_attn_score.data_ptr<DType>();
  } else if constexpr (OutputMuAppliedAttnScoreSwitch == 3) {
    gdata.mu_softmax_applied_unnormalized_attn_score =
        mu_softmax_applied_unnormalized_attn_score.data_ptr<DType>();
    gdata.normalized_attn_score = normalized_attn_score.data_ptr<DType>();
  } else {
    assert(0 && "invalid OutputMuAppliedAttnScoreSwitch");
  }

  // Configure kernel launch parameters. From _gatExpLeakyReluSumKernel
  // configurations in
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl
  int nthrs_x = 32;
  int nthrs_y = 1;
  int nblks_x = (num_heads + nthrs_x - 1) / (nthrs_x);
  int64_t incsr_num_rows_or_edges = incsr_rowptr_or_indices.numel() - 1;
  int nblks_y = std::min(incsr_num_rows_or_edges, MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);

  if (EdgeParallelFlag) {
    _hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel<
        Idx, DType, false, true, true, false, OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_rowptr_or_indices.data_ptr<Idx>(),
            incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes_or_relptr.data_ptr<Idx>(), incsr_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
    _hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel_stage_2<
        Idx, DType, false, true, true, false, OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_rowptr_or_indices.data_ptr<Idx>(),
            incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes_or_relptr.data_ptr<Idx>(), incsr_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
  } else {
    _hgtEdgeSoftmaxAccumStageOnlyKernel<Idx, DType, false, true, false, false,
                                        OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_rowptr_or_indices.data_ptr<Idx>(),
            incsr_col_idx.data_ptr<Idx>(),
            incsr_reltypes_or_relptr.data_ptr<Idx>(), incsr_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
  }
}

void full_graph_edge_softmax_ops(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& unnormalized_attn_score, at::Tensor& mu,
    at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  _full_graph_edge_softmax_ops<int64_t, float, 3, false>(
      row_ptr, col_idx, eids, reltypes, unnormalized_attn_score, mu,
      edgesoftmax_sum_per_node, mu_softmax_applied_unnormalized_attn_score,
      normalized_attn_score);
}

// this function only calculates edge softmax sum at each destination node.
void full_graph_edge_softmax_only_accumu_stage_ops(
    at::Tensor& incsr_rowptr, at::Tensor& incsr_col_idx, at::Tensor& incsr_eids,
    at::Tensor& incsr_reltypes, at::Tensor& unnormalized_attn_score,
    at::Tensor& mu, at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score) {
  // using _hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]
  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that does only the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
  at::Tensor dummy_tensor;
  _full_graph_edge_softmax_ops<int64_t, float, 1, false>(
      incsr_rowptr, incsr_col_idx, incsr_eids, incsr_reltypes,
      unnormalized_attn_score, mu, edgesoftmax_sum_per_node,
      mu_softmax_applied_unnormalized_attn_score, dummy_tensor);
}

}  // namespace IntegratedCSR

void full_graph_edge_softmax_ops_separate_coo(
    at::Tensor& row_indices, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes_ptr, at::Tensor& unnormalized_attn_score,
    at::Tensor& mu, at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  IntegratedCSR::_full_graph_edge_softmax_ops<int64_t, float, 3, true>(
      row_indices, col_idx, eids, reltypes_ptr, unnormalized_attn_score, mu,
      edgesoftmax_sum_per_node, mu_softmax_applied_unnormalized_attn_score,
      normalized_attn_score);
}

}  // namespace FwProp
namespace BckProp {
namespace IntegratedCSR {

// from HGTBackPropGradientSMAFusion in
// [[hetero_edgesoftmax/include/DGLHackKernel/OpPrototyping/HGTProtoOps.h]]
template </*int XPU, */ typename Idx, typename DType>
void HGTBackPropGradientSMAFusionExperimental(
    // GraphRef graph,
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_eids,
    at::Tensor& csr_reltypes, at::Tensor& grad_sm_first_stage,
    at::Tensor& grad_a, at::Tensor& grad_t_neighbour, at::Tensor& message,
    at::Tensor& sigmas) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // LOG(INFO) << "Calling implementation of rgn layer 1 forward";
  // assert(csr.IsSortedByEdgeType_CPU());
  // typedef int32_t Idx;
  // typedef float DType;
  // auto csr = graph->GetCsrSortedByEdgeType(false);
  // auto ranges = csr[0];
  // auto ids = csr[1];
  // auto eids = csr[2];
  // auto type_ids = csr[3];
  auto range_data = csr_rowptr.data_ptr<Idx>();
  auto ids_data = csr_col_idx.data_ptr<Idx>();
  // auto eids_data = static_cast<Idx*>(thrust::raw_pointer_cast(eids);
  auto eids_data = csr_eids.data_ptr<Idx>();
  auto typeids_data = csr_reltypes.data_ptr<Idx>();
  auto grad_sm_first_stage_data = grad_sm_first_stage.data_ptr<DType>();
  auto grad_a_data = grad_a.data_ptr<DType>();
  auto grad_t_neighbour_data = grad_t_neighbour.data_ptr<DType>();
  auto message_data = message.data_ptr<DType>();
  auto sigmas_data = sigmas.data_ptr<DType>();

  // print_dims(hidden);
  // print_dims(weight);
  // print_dims(norm);
  // print_dims(ret);
  // Idx num_nodes = ranges->shape[0] - 1;
  // Idx num_edges = eids->shape[0];
  Idx num_nodes = csr_rowptr.numel() - 1;
  Idx num_edges = csr_col_idx.numel();
  Idx num_heads = grad_sm_first_stage.size(2);
  Idx feat_dim_per_head = grad_sm_first_stage.size(3);
  Idx n_rel_types = grad_sm_first_stage.size(1);
  int nblks = num_nodes;
  int nthrs = num_heads * feat_dim_per_head;
  // auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t1 =
  //     std::chrono::high_resolution_clock::now();
  HGTBackwardGradientSmFirstPartImpl<Idx, DType><<<nblks, nthrs, 0, stream>>>(
      range_data, ids_data, eids_data, typeids_data, grad_sm_first_stage_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "HGTBackwardGradientSmFirstPartImpl time: "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //     t1).count()
  //     << " ms" << std::endl;

  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t1_kernel2 =
  //     std::chrono::high_resolution_clock::now();
  HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_a_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2_kernel2 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "HGTBackwardGradientAImpl time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  t2_kernel2 - t1_kernel2)
  //                  .count()
  //           << " ms" << std::endl;

  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t1_kernel3 =
  //     std::chrono::high_resolution_clock::now();

  HGTBackwardFusedGradientSmFirstPartGradientAImpl<Idx, DType>
      <<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_a_data,
          grad_sm_first_stage_data, grad_t_neighbour_data, message_data,
          sigmas_data, num_nodes, num_heads, feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2_kernel3 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "HGTBackwardFusedGradientSmFirstPartGradientAImpl time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  t2_kernel3 - t1_kernel3)
  //                  .count()
  //           << " ms" << std::endl;
}

template <typename Idx, typename DType,
          int AttnScoreUseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation_and_edge_softmax(
    at::Tensor& outcsr_rowptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_reltypes, at::Tensor& outcsr_eids, at::Tensor& message,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& unnormalized_attn_score,
    at::Tensor& mu, at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score, at::Tensor& out, at::Tensor& gradout,
    at::Tensor& grad_attn_score, at::Tensor& grad_message,
    at::Tensor& grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // preparing gdata_attn
  BackwardHGTAttnScoreData<Idx, DType, AttnScoreUseMuAppliedAttnScoreSwitch>
      gdata_attn;
  Idx num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / num_heads;
  // FIXME: remove repeated struct members
  gdata_attn.num_heads = num_heads;
  assert(gdata_attn.num_heads == message.size(message.ndimension() - 2) &&
         "expecting message.size[-2] to be num_heads but turned out not");
  gdata_attn.message_src_xlen =
      message.size(grad_mu.ndimension() - 1) * gdata_attn.num_heads;
  gdata_attn.message_src = message.data_ptr<DType>();
  gdata_attn.eids = outcsr_eids.data_ptr<Idx>();
  gdata_attn.grad_attn_score = grad_attn_score.data_ptr<DType>();
  gdata_attn.unnormalized_attn_score =
      unnormalized_attn_score.data_ptr<DType>();

  gdata_attn.out = out.data_ptr<DType>();
  gdata_attn.grad_out = gradout.data_ptr<DType>();
  gdata_attn.mu = mu.data_ptr<DType>();
  gdata_attn.grad_mu = grad_mu.data_ptr<DType>();
  if constexpr (AttnScoreUseMuAppliedAttnScoreSwitch == 0) {
    gdata_attn.edgesoftmax_sum_per_node =
        edgesoftmax_sum_per_node.data_ptr<DType>();
  } else if constexpr (AttnScoreUseMuAppliedAttnScoreSwitch == 1) {
    gdata_attn.edgesoftmax_sum_per_node =
        edgesoftmax_sum_per_node.data_ptr<DType>();
    gdata_attn.mu_softmax_applied_unnormalized_attn_score =
        edgesoftmax_sum_per_node.data_ptr<DType>();
  } else if constexpr (AttnScoreUseMuAppliedAttnScoreSwitch == 2) {
    gdata_attn.normalized_attn_score = normalized_attn_score.data_ptr<DType>();
  } else {
    assert(false && "AttnScoreUseMuAppliedAttnScoreSwitch must be 0, 1 or 2");
  }

  // preparing gdata_msg
  // BackwardHGTMessageData<Idx, DType, MessageUseMuAppliedAttnScoreSwitch>
  // gdata_msg; there are only one additional field to attn_score's struct
  // needed, i.e., gdata_msg.grad_message_src = grad_message.data_ptr<DType>();
  // Meanwhile, gdata_msg.grad_out is the gradient of output node feature and
  // thus identical with gdata_attn.gradout if constexpr
  // (MessageUseMuAppliedAttnScoreSwitch < 0 ||
  //               MessageUseMuAppliedAttnScoreSwitch > 2) {
  //   assert(false && "MessageUseMuAppliedAttnScoreSwitch must be 0, 1 or 2");
  // }

  // preparing kernel launch configuration
  int nthrs_x = SeastarFindNumThreads(num_heads, 64);
  int nthrs_y = SeastarFindNumThreads(
      gdata_attn.message_src_xlen,
      MAX_NTHRS / nthrs_x);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_rowptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);

  // gdata_msg.grad_message_src = grad_message.data_ptr<DType>();
  _hgtAttnAndMessageSrcFusedBckKernel<Idx, DType, false, true, false,
                                      AttnScoreUseMuAppliedAttnScoreSwitch>
      <<<nblks, nthrs, 0, stream>>>(
          gdata_attn, grad_message.data_ptr<DType>(),
          outcsr_rowptr.data_ptr<Idx>(), outcsr_col_idx.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          /*no need when !CompactAsOfNodeFlag*/ nullptr,
          /*no need when !CompactAsOfNodeFlag*/ nullptr, num_relations);
}

template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor& outcsr_rowptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_reltypes, at::Tensor& outcsr_eids,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& unnormalized_attn_score,
    at::Tensor& mu, at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score, at::Tensor& gradout,
    at::Tensor& grad_message) {
  // using _hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel
  // that based on _fusedGatBackwardGradFeatSrc whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // preparing gdata
  BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads =
      normalized_attn_score.size(normalized_attn_score.ndimension() - 1);
  Idx num_relations = mu.numel() / num_heads;
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "message_mean_aggregation"
  //             << std::endl;
  // }

  gdata.num_heads = num_heads;
  assert(gdata.num_heads == grad_message.size(grad_message.ndimension() - 2) &&
         "assuming num_heads is the same for grad_message but turned out not");
  gdata.message_src_xlen =
      grad_message.size(grad_message.ndimension() - 1) * gdata.num_heads;
  gdata.eids = outcsr_eids.data_ptr<Idx>();
  gdata.grad_message_src = grad_message.data_ptr<DType>();
  gdata.grad_out = gradout.data_ptr<DType>();

  if constexpr (UseMuAppliedAttnScoreSwitch == 0) {
    gdata.mu = mu.data_ptr<DType>();
    gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
    gdata.unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>();
  } else if constexpr (UseMuAppliedAttnScoreSwitch == 1) {
    gdata.mu_softmax_applied_unnormalized_attn_score =
        mu_softmax_applied_unnormalized_attn_score.data_ptr<DType>();
    gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
    gdata.unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>();
    gdata.mu = mu.data_ptr<DType>();
  } else if constexpr (UseMuAppliedAttnScoreSwitch == 2) {
    gdata.normalized_attn_score = normalized_attn_score.data_ptr<DType>();
  } else {
    assert(false && "UseMuAppliedAttnScoreSwitch must be 0, 1 or 2");
  }

  // kernel parameter configurations
  int nthrs_x = SeastarFindNumThreads(num_heads, 64);
  int nthrs_y = SeastarFindNumThreads(
      gdata.message_src_xlen,
      MAX_NTHRS / nthrs_x);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_rowptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);

  _hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel<
      Idx, DType, false, true, false, UseMuAppliedAttnScoreSwitch>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, outcsr_rowptr.data_ptr<Idx>(), outcsr_col_idx.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          /*no need unless CompactAsOfNode*/ nullptr,
          /*no need unless CompactAsOfNode*/ nullptr, num_relations);
}

void full_graph_message_mean_aggregation(
    at::Tensor& outcsr_rowptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_reltypes, at::Tensor& outcsr_eids,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& normalized_attn_score,
    at::Tensor& gradout, at::Tensor& grad_message) {
  at::Tensor dummy_tensor;
  _full_graph_message_mean_aggregation<int64_t, float, 2>(
      outcsr_rowptr, outcsr_col_idx, outcsr_reltypes, outcsr_eids,
      edgesoftmax_sum_per_node, dummy_tensor, dummy_tensor, dummy_tensor,
      normalized_attn_score, gradout, grad_message);
}

void full_graph_hetero_attention_ops(at::Tensor& row_ptr, at::Tensor& col_idx,
                                     at::Tensor& eids, at::Tensor& reltypes,
                                     at::Tensor& weight,
                                     at::Tensor& applied_klinear_node_features,
                                     at::Tensor& applied_qlinear_node_features,
                                     at::Tensor& gradout,
                                     at::Tensor& grad_weight,
                                     at::Tensor& grad_k, at::Tensor& grad_q) {
  // we need to implement a fused kernel based on back prop of RGNN
  // inner_product and back prop of W*t via RGNN relational_matmul

  assert(0 && "the performant implementation not done yet");
}

template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch>
void _full_graph_edge_softmax_ops(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& message,
    at::Tensor& unnormalized_attn_score, at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score, at::Tensor& out, at::Tensor& gradout,
    at::Tensor& mu, at::Tensor& grad_attn_score, at::Tensor& grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // preparing gdata
  BackwardHGTAttnScoreData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / num_heads;
  gdata.num_heads = num_heads;
  assert(gdata.num_heads == message.size(message.ndimension() - 2) &&
         "expecting message.size[-2] to be num_heads but turned out not");
  gdata.message_src_xlen =
      message.size(grad_mu.ndimension() - 1) * gdata.num_heads;
  gdata.message_src = message.data_ptr<DType>();
  gdata.eids = outcsr_eids.data_ptr<Idx>();
  gdata.grad_attn_score = grad_attn_score.data_ptr<DType>();
  gdata.unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>();

  gdata.out = out.data_ptr<DType>();
  gdata.grad_out = gradout.data_ptr<DType>();
  gdata.mu = mu.data_ptr<DType>();
  gdata.grad_mu = grad_mu.data_ptr<DType>();
  if constexpr (OutputMuAppliedAttnScoreSwitch == 0) {
    gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
  } else if constexpr (OutputMuAppliedAttnScoreSwitch == 1) {
    gdata.edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>();
    gdata.mu_softmax_applied_unnormalized_attn_score =
        edgesoftmax_sum_per_node.data_ptr<DType>();
  } else if constexpr (OutputMuAppliedAttnScoreSwitch == 2) {
    gdata.normalized_attn_score = normalized_attn_score.data_ptr<DType>();
  } else {
    assert(false && "OutputMuAppliedAttnScoreSwitch must be 0, 1 or 2");
  }

  // preparing kernel launch configuration
  int nthrs_x = SeastarFindNumThreads(num_heads, 64);
  int nthrs_y = SeastarFindNumThreads(
      gdata.message_src_xlen,
      MAX_NTHRS / nthrs_x);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  _hgtEdgeSoftmaxAccumStageOnlyBackwardKernel<Idx, DType, false, true, false,
                                              OutputMuAppliedAttnScoreSwitch>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, outcsr_row_ptr.data_ptr<Idx>(), outcsr_col_idx.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          /*no need when !CompactAsOfNodeFlag*/ nullptr,
          /*no need when !CompactAsOfNodeFlag*/ nullptr, num_relations);
}

void full_graph_edge_softmax_ops(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& message,
    at::Tensor& unnormalized_attn_score, at::Tensor& normalized_attn_score,
    at::Tensor& out, at::Tensor& gradout, at::Tensor& mu,
    at::Tensor& grad_attn_score, at::Tensor& grad_mu) {
  at::Tensor dummy_tensor;
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  _full_graph_edge_softmax_ops<int64_t, float, 2>(
      outcsr_row_ptr, outcsr_col_idx, outcsr_eids, outcsr_reltypes, message,
      unnormalized_attn_score, dummy_tensor, dummy_tensor,
      normalized_attn_score, out, gradout, mu, grad_attn_score, grad_mu);
}

void full_graph_edge_softmax_only_accumu_stage_ops(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& message,
    at::Tensor& unnormalized_attn_score, at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score, at::Tensor& out,
    at::Tensor& gradout, at::Tensor& mu, at::Tensor& grad_attn_score,
    at::Tensor& grad_mu) {
  // using _hgtEdgeSoftmaxAccumStageOnlyBackwardKernel based on
  // _fusedGatBackwardGradElEr whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that only do the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
  at::Tensor dummy_tensor;
  _full_graph_edge_softmax_ops<int64_t, float, 1>(
      outcsr_row_ptr, outcsr_col_idx, outcsr_eids, outcsr_reltypes, message,
      unnormalized_attn_score, edgesoftmax_sum_per_node,
      mu_softmax_applied_unnormalized_attn_score, dummy_tensor, out, gradout,
      mu, grad_attn_score, grad_mu);
}

}  // namespace IntegratedCSR
}  // namespace BckProp
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET
