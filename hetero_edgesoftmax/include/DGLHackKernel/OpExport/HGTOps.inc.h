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
// #include "DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

// TODO: add torch tensor version of HGTForwardImpl from
// [[hetero_edgesoftmax/include/DGLHackKernel/OpPrototyping/HGTProtoOps.h]]

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {

// this function 1) (accumulation stage) calculates edge softmax sum at each
// destination node, and 2) (scatter stage) normalize the attention score by the
// sum at each edge
template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch,
          bool EdgeParallelFlag>
void _full_graph_edge_softmax_ops(
    at::Tensor& incsr_or_sep_coo_rowptr_or_indices,
    at::Tensor& incsr_or_sep_coo_col_idx, at::Tensor& incsr_or_sep_coo_eids,
    at::Tensor& incsr_or_sep_coo_reltypes_or_relptr,
    at::Tensor& unnormalized_attn_score, at::Tensor& mu,
    at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score) {
  // using HET__hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
  // There is an existing implementation with tricky API in
  // hetero_edgesoftmax/include/EdgeSoftmax_1/EdgeSoftmaxCSR.h
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // Configure kernel parameters structure
  HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata;
  gdata.num_heads =
      unnormalized_attn_score.size(unnormalized_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::FwProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / gdata.num_heads;

  gdata.eids = incsr_or_sep_coo_eids.data_ptr<Idx>();
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

  // Type 1 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
  // head -> blockIdx.x * blockDim.x + threadIdx.x;
  // edge|node -> blockIdx.y * blockDim.y + threadIdx.y;

  int nthrs_x = 1;
  int nthrs_y = 32;
  int nblks_x = (gdata.num_heads + nthrs_x - 1) / (nthrs_x);
  int64_t incsr_or_sep_coo_num_rows_or_edges =
      incsr_or_sep_coo_rowptr_or_indices.numel() - 1;
  int nblks_y =
      std::min(ceil_div(incsr_or_sep_coo_num_rows_or_edges, (int64_t)nthrs_y),
               MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);

  if constexpr (EdgeParallelFlag) {
    // use separate coo instead of in csr
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel<
        Idx, DType, false, true, true, false, OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_or_sep_coo_rowptr_or_indices.data_ptr<Idx>(),
            incsr_or_sep_coo_col_idx.data_ptr<Idx>(),
            incsr_or_sep_coo_reltypes_or_relptr.data_ptr<Idx>(),
            incsr_or_sep_coo_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel_stage_2<
        Idx, DType, false, true, true, false, OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_or_sep_coo_rowptr_or_indices.data_ptr<Idx>(),
            incsr_or_sep_coo_col_idx.data_ptr<Idx>(),
            incsr_or_sep_coo_reltypes_or_relptr.data_ptr<Idx>(),
            incsr_or_sep_coo_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
  } else {
    // use in csr
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel<
        Idx, DType, false, true, false, false, OutputMuAppliedAttnScoreSwitch>
        <<<nblks, nthrs, 0, stream>>>(
            gdata, incsr_or_sep_coo_rowptr_or_indices.data_ptr<Idx>(),
            incsr_or_sep_coo_col_idx.data_ptr<Idx>(),
            incsr_or_sep_coo_reltypes_or_relptr.data_ptr<Idx>(),
            incsr_or_sep_coo_num_rows_or_edges,
            /*no need when !CompactAsOfNode*/ nullptr,
            /*no need when !CompactAsOfNode*/ nullptr, num_relations);
  }
}

namespace IntegratedCSR {
template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor& incsr_rowptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_reltypes, at::Tensor& incsr_eids,
    at::Tensor& edge_messages, at::Tensor& edge_attn_score,
    at::Tensor& edgesoftmax_sum_per_node, at::Tensor& mu, at::Tensor& ret) {
  // using HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum based on
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
  gdata.num_heads = edge_attn_score.size(edge_attn_score.ndimension() - 1);
  Idx num_relations = mu.numel() / gdata.num_heads;
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
  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
  // original seastar schedule to allow reduction within the warp, i.e., along
  // x-axis
  int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
  int nthrs_x = SeastarFindNumThreads(
      gdata.message_out_dim,
      MAX_NTHRS /
          nthrs_y);  // NB: message_out_dim is the total dim, and the number of
                     // elements for each head is message_out_dim//num_heads
  int nblks_x = 1;
  int nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  // printf("nblks_x=%d, nblks_y=%d, nthrs_x=%d, nthrs_y=%d\n", nblks_x,
  // nblks_y,
  //       nthrs_x, nthrs_y);
  HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum<
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

void full_graph_edge_softmax_ops(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& unnormalized_attn_score, at::Tensor& mu,
    at::Tensor& edgesoftmax_sum_per_node,
    at::Tensor& mu_softmax_applied_unnormalized_attn_score,
    at::Tensor& normalized_attn_score) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  HET::TorchExport::HGT::FwProp::_full_graph_edge_softmax_ops<int64_t, float, 3,
                                                              false>(
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
  // using HET__hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]
  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that does only the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
  at::Tensor dummy_tensor;
  HET::TorchExport::HGT::FwProp::_full_graph_edge_softmax_ops<int64_t, float, 1,
                                                              false>(
      incsr_rowptr, incsr_col_idx, incsr_eids, incsr_reltypes,
      unnormalized_attn_score, mu, edgesoftmax_sum_per_node,
      mu_softmax_applied_unnormalized_attn_score, dummy_tensor);
}

}  // namespace IntegratedCSR

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
  HET_HGTBackwardGradientSmFirstPartImpl<Idx, DType>
      <<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data,
          grad_sm_first_stage_data, grad_t_neighbour_data, message_data,
          sigmas_data, num_nodes, num_heads, feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout
  //     << "HET_HGTBackwardGradientSmFirstPartImpl time: "
  //     << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
  //     t1).count()
  //     << " ms" << std::endl;

  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t1_kernel2 =
  //     std::chrono::high_resolution_clock::now();
  HET_HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_a_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2_kernel2 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "HET_HGTBackwardGradientAImpl time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  t2_kernel2 - t1_kernel2)
  //                  .count()
  //           << " ms" << std::endl;

  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t1_kernel3 =
  //     std::chrono::high_resolution_clock::now();

  HET_HGTBackwardFusedGradientSmFirstPartGradientAImpl<Idx, DType>
      <<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_a_data,
          grad_sm_first_stage_data, grad_t_neighbour_data, message_data,
          sigmas_data, num_nodes, num_heads, feat_dim_per_head, n_rel_types);
  // cuda_err_chk(cudaPeekAtLastError());
  // cuda_err_chk(cudaDeviceSynchronize());
  // std::chrono::high_resolution_clock::time_point t2_kernel3 =
  //     std::chrono::high_resolution_clock::now();
  // std::cout << "HET_HGTBackwardFusedGradientSmFirstPartGradientAImpl time: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  t2_kernel3 - t1_kernel3)
  //                  .count()
  //           << " ms" << std::endl;
}

// adapted from the BckProp::_full_graph_edge_softmax_ops, the wrapper function
// of HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel
template <typename Idx, typename DType>
void full_graph_EdgeSoftmax_eNorm_to_UnNormalizedAttnScore(
    at::Tensor& incsr_row_ptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_eids, at::Tensor& incsr_reltypes,
    at::Tensor& unnormalized_attn_score, at::Tensor& normalized_attn_score,
    at::Tensor& grad_normalized_attn_score, at::Tensor& mu,
    at::Tensor& grad_unnormalized_attn_score, at::Tensor& grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // preparing gdata
  BackwardNormToUnNormalizedAttnScoreData<Idx, DType> gdata;
  gdata.num_heads = grad_normalized_attn_score.size(
      grad_normalized_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / gdata.num_heads;
  gdata.eids = incsr_eids.data_ptr<Idx>();
  gdata.grad_normalized_attn_score =
      grad_normalized_attn_score.data_ptr<DType>();
  gdata.unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>();
  gdata.grad_unnormalized_attn_score =
      grad_unnormalized_attn_score.data_ptr<DType>();

  gdata.mu = mu.data_ptr<DType>();
  gdata.grad_mu = grad_mu.data_ptr<DType>();

  // based on OutputMuAppliedAttnScoreSwitch==2 in
  // HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_edge_softmax_ops
  gdata.normalized_attn_score = normalized_attn_score.data_ptr<DType>();

  // preparing kernel launch configuration
  // NB: Type 1 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
  // head -> blockIdx.x * blockDim.x + threadIdx.x;
  // node -> blockIdx.y * blockDim.y + threadIdx.y;
  int nthrs_y = 32;
  int nthrs_x = 1;
  int nblks_x = (gdata.num_heads + nthrs_x - 1) / (nthrs_x);
  int nblks_y = std::min(ceil_div(incsr_row_ptr.numel() - 1, (int64_t)nthrs_y),
                         MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  HET_EdgeSoftmaxENormToUnNormalizedAttnScoreBackwardKernel<Idx, DType, true,
                                                            false>
      <<<nblks, nthrs, 0, stream>>>(
          gdata, incsr_row_ptr.data_ptr<Idx>(), incsr_col_idx.data_ptr<Idx>(),
          incsr_reltypes.data_ptr<Idx>(), incsr_row_ptr.numel() - 1,
          /*no need when !CompactAsOfNodeFlag*/ nullptr,
          /*no need when !CompactAsOfNodeFlag*/ nullptr, num_relations);
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
  gdata_attn.num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / gdata_attn.num_heads;
  // NB: grad_attn_score structure is kept and the unique items from
  // grad_message structure is passed into kernel as individual parameters one
  // by one
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

  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  int nthrs_y = SeastarFindNumThreads(gdata_attn.num_heads, 64);
  int nthrs_x = SeastarFindNumThreads(
      gdata_attn.message_src_xlen,
      MAX_NTHRS / nthrs_y);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_rowptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);

  // gdata_msg.grad_message_src = grad_message.data_ptr<DType>();
  HET__hgtAttnAndMessageSrcFusedBckKernel<Idx, DType, false, true, false,
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
  // using
  // HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel that
  // based on _fusedGatBackwardGradFeatSrc whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // preparing gdata
  BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata;
  gdata.num_heads =
      normalized_attn_score.size(normalized_attn_score.ndimension() - 1);
  Idx num_relations = mu.numel() / gdata.num_heads;
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "message_mean_aggregation"
  //             << std::endl;
  // }

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
  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
  int nthrs_x = SeastarFindNumThreads(
      gdata.message_src_xlen,
      MAX_NTHRS / nthrs_y);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_rowptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);

  HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel<
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
  gdata.num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  // if (num_heads <= 1) {
  //   std::cout << "Warning: num_heads <= 1 in "
  //                "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
  //                "edge_softmax_ops"
  //             << std::endl;
  // }
  Idx num_relations = mu.numel() / gdata.num_heads;
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
  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
  // original seastar schedule to allow reduction within the warp, i.e., along
  // x-axis
  int nthrs_y = SeastarFindNumThreads(gdata.num_heads, 64);
  int nthrs_x = SeastarFindNumThreads(
      gdata.message_src_xlen,
      MAX_NTHRS / nthrs_y);  // NB: message_src_xlen is the total dimension
                             // whereas each head gets message_src_xlen //
                             // num_heads number of elements
  int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
  int nblks_x = 1;
  int nblks_y = std::min(outcsr_num_rows, MAX_NBLKS);
  const dim3 nthrs(nthrs_x, nthrs_y);
  const dim3 nblks(nblks_x, nblks_y);
  HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel<
      Idx, DType, false, true, false, OutputMuAppliedAttnScoreSwitch>
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

// TODO: this function is not exported yet. Merge this with the
// full_graph_edge_softmax_ops API with an additional integer indicating the
// template int MuAppliedAttnScoreSwitch
// void full_graph_edge_softmax_only_accumu_stage_ops(
//     at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
//     at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor&
//     message, at::Tensor& unnormalized_attn_score, at::Tensor&
//     edgesoftmax_sum_per_node, at::Tensor&
//     mu_softmax_applied_unnormalized_attn_score, at::Tensor& out, at::Tensor&
//     gradout, at::Tensor& mu, at::Tensor& grad_attn_score, at::Tensor&
//     grad_mu) {
//   // using HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel based on
//   // _fusedGatBackwardGradElEr whose driver code is
//   // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
//   // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

//   // NB: call the partial specialized version of _full_graph_edge_softmax_ops
//   // that only do the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
//   at::Tensor dummy_tensor;
//   _full_graph_edge_softmax_ops<int64_t, float, 1>(
//       outcsr_row_ptr, outcsr_col_idx, outcsr_eids, outcsr_reltypes, message,
//       unnormalized_attn_score, edgesoftmax_sum_per_node,
//       mu_softmax_applied_unnormalized_attn_score, dummy_tensor, out, gradout,
//       mu, grad_attn_score, grad_mu);
// }

}  // namespace IntegratedCSR
}  // namespace BckProp
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET

using namespace HET::TorchExport;
TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
  // HGT CSR Declaration
  m.def("backward_hgt_full_graph_message_mean_aggregation_csr",
        HGT::BckProp::IntegratedCSR::full_graph_message_mean_aggregation);
  m.def("hgt_full_graph_message_mean_aggregation_csr",
        HGT::FwProp::IntegratedCSR::full_graph_message_mean_aggregation);
  m.def("hgt_full_graph_edge_softmax_ops_csr",
        HGT::FwProp::IntegratedCSR::full_graph_edge_softmax_ops);
  m.def("backward_hgt_full_graph_edge_softmax_ops_csr",
        HGT::BckProp::IntegratedCSR::full_graph_edge_softmax_ops);
  m.def("backward_hgt_full_graph_enorm_to_unnormalized_attn_score_csr",
        HGT::BckProp::IntegratedCSR::
            full_graph_EdgeSoftmax_eNorm_to_UnNormalizedAttnScore<int64_t,
                                                                  float>);
}
