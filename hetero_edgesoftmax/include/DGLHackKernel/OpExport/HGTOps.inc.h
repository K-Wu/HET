#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "DGLHackKernel/HGT/HGTBackwardExperimental.cu.h"
#include "DGLHackKernel/HGT/HGTBackwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTForwardKernels.cu.h"
#include "ThreadingGridsBlocksSchedules.h"

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
    at::Tensor &incsr_or_sep_coo_row_ptrs_or_indices,
    at::Tensor &incsr_or_sep_coo_col_indices, at::Tensor &incsr_or_sep_coo_eids,
    at::Tensor &incsr_or_sep_coo_reltypes_or_relptrs,
    at::Tensor &unnormalized_attn_score, at::Tensor &mu,
    at::Tensor &edgesoftmax_sum_per_node,
    at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score) {
  // using HET__hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
  // There is an existing implementation with tricky API in
  // hetero_edgesoftmax/include/EdgeSoftmax_1/EdgeSoftmaxCSR.h
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Configure kernel parameters structure
  HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata{
      .num_heads = unnormalized_attn_score.size(
          unnormalized_attn_score.ndimension() - 1),
      .eids = incsr_or_sep_coo_eids.data_ptr<Idx>(),
      .mu = mu.data_ptr<DType>(),
      .unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>(),
      .edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>()};

  Idx num_relations = mu.numel() / gdata.num_heads;
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
  int64_t incsr_or_sep_coo_num_rows_or_edges =
      incsr_or_sep_coo_row_ptrs_or_indices.numel() - 1;
  auto [nblks, nthrs] =
      get_type1_schedule(gdata.num_heads, incsr_or_sep_coo_num_rows_or_edges);

  if constexpr (EdgeParallelFlag) {
    // use separate coo instead of in csr
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel<
        Idx, DType, CompactAsOfNodeKind::Disabled, true, true, false,
        OutputMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
        gdata, incsr_or_sep_coo_row_ptrs_or_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_col_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_reltypes_or_relptrs.data_ptr<Idx>(),
        incsr_or_sep_coo_num_rows_or_edges, {}, num_relations);
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel_edgeparallel_stage_2<
        Idx, DType, CompactAsOfNodeKind::Disabled, true, true, false,
        OutputMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
        gdata, incsr_or_sep_coo_row_ptrs_or_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_col_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_reltypes_or_relptrs.data_ptr<Idx>(),
        incsr_or_sep_coo_num_rows_or_edges, {}, num_relations);
  } else {
    // use in csr
    HET__hgtEdgeSoftmaxAccumStageOnlyKernel<
        Idx, DType, CompactAsOfNodeKind::Disabled, true, false, false,
        OutputMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
        gdata, incsr_or_sep_coo_row_ptrs_or_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_col_indices.data_ptr<Idx>(),
        incsr_or_sep_coo_reltypes_or_relptrs.data_ptr<Idx>(),
        incsr_or_sep_coo_num_rows_or_edges, {}, num_relations);
  }
}

namespace IntegratedCSR {
template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor &incsr_row_ptrs, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_reltypes, at::Tensor &incsr_eids,
    at::Tensor &edge_messages, at::Tensor &edge_attn_score,
    at::Tensor &edgesoftmax_sum_per_node, at::Tensor &mu, at::Tensor &ret) {
  // using HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum based on
  // _gatSumProdZipDivKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]] NB:
  // based on (vertex-centric) _gatSumProdZipDivKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]] or
  // (edge-centric) _gatSumProdZipDivKernel_edge_parallel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATKernelsSeparateCOO.cu.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // configure parameter struct

  int num_heads = edge_attn_score.size(edge_attn_score.ndimension() - 1);
  HgtDstOutData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata{
      .num_heads = num_heads,
      .message_out_dim =
          edge_messages.size(edge_messages.ndimension() - 1) * num_heads,
      .eids = incsr_eids.data_ptr<Idx>(),
      .edgesoftmax_sum_per_node = edgesoftmax_sum_per_node.data_ptr<DType>(),
      .message = edge_messages.data_ptr<DType>(),
      .ret = ret.data_ptr<DType>()};
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

  Idx num_relations = mu.numel() / gdata.num_heads;
  assert(gdata.num_heads ==
             edge_messages.size(edge_messages.ndimension() - 2) &&
         "assuming edge_messages[-2] to be num_heads but failed");

  Idx incsr_num_rows = incsr_row_ptrs.numel() - 1;

  // Configure kernel launch parameters.
  // NB: updated to Type 2 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-a90053897bc12f11e78835acb7eb0539b67430a2cd7da43d586dab113fdeafefL373-R385
  // head -> threadIdx.y
  // node -> blockIdx.y
  // feat_idx -> blockIdx.x * blockDim.x + threadIdx.x
  // threadIdx.x and threadIdx.y and only this pair is exchanged compared with
  // original seastar schedule to allow reduction within the warp, i.e., along
  // x-axis
  auto [nblks, nthrs] = get_type2_schedule(
      gdata.num_heads, gdata.message_out_dim, incsr_num_rows);
  HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSum<
      Idx, DType, CompactAsOfNodeKind::Disabled, true, false, false,
      UseMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
      gdata, incsr_row_ptrs.data_ptr<Idx>(), incsr_col_indices.data_ptr<Idx>(),
      incsr_reltypes.data_ptr<Idx>(), incsr_num_rows, {}, num_relations);
}

void full_graph_message_mean_aggregation(
    at::Tensor &incsr_row_ptrs, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_reltypes, at::Tensor &incsr_eids,
    at::Tensor &edge_messages, at::Tensor &edge_attn_score,
    at::Tensor &edgesoftmax_sum_per_node, at::Tensor &mu, at::Tensor &ret) {
  _full_graph_message_mean_aggregation<int64_t, float, 1>(
      incsr_row_ptrs, incsr_col_indices, incsr_reltypes, incsr_eids,
      edge_messages, edge_attn_score, edgesoftmax_sum_per_node, mu, ret);
}

void full_graph_edge_softmax_ops(
    at::Tensor &row_ptr, at::Tensor &col_indices, at::Tensor &eids,
    at::Tensor &reltypes, at::Tensor &unnormalized_attn_score, at::Tensor &mu,
    at::Tensor &edgesoftmax_sum_per_node,
    at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  HET::TorchExport::HGT::FwProp::_full_graph_edge_softmax_ops<int64_t, float, 3,
                                                              false>(
      row_ptr, col_indices, eids, reltypes, unnormalized_attn_score, mu,
      edgesoftmax_sum_per_node, mu_softmax_applied_unnormalized_attn_score,
      normalized_attn_score);
}

// this function only calculates edge softmax sum at each destination node.
void full_graph_edge_softmax_only_accumu_stage_ops(
    at::Tensor &incsr_row_ptrs, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unnormalized_attn_score, at::Tensor &mu,
    at::Tensor &edgesoftmax_sum_per_node,
    at::Tensor &mu_softmax_applied_unnormalized_attn_score) {
  // using HET__hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]
  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that does only the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
  at::Tensor dummy_tensor;
  HET::TorchExport::HGT::FwProp::_full_graph_edge_softmax_ops<int64_t, float, 1,
                                                              false>(
      incsr_row_ptrs, incsr_col_indices, incsr_eids, incsr_reltypes,
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
    at::Tensor &csr_row_ptrs, at::Tensor &csr_col_indices, at::Tensor &csr_eids,
    at::Tensor &csr_reltypes, at::Tensor &grad_sm_first_stage,
    at::Tensor &grad_a, at::Tensor &grad_t_neighbour, at::Tensor &message,
    at::Tensor &sigmas) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  auto range_data = csr_row_ptrs.data_ptr<Idx>();
  auto ids_data = csr_col_indices.data_ptr<Idx>();
  auto eids_data = csr_eids.data_ptr<Idx>();
  auto typeids_data = csr_reltypes.data_ptr<Idx>();
  auto grad_sm_first_stage_data = grad_sm_first_stage.data_ptr<DType>();
  auto grad_a_data = grad_a.data_ptr<DType>();
  auto grad_t_neighbour_data = grad_t_neighbour.data_ptr<DType>();
  auto message_data = message.data_ptr<DType>();
  auto sigmas_data = sigmas.data_ptr<DType>();

  Idx num_nodes = csr_row_ptrs.numel() - 1;
  Idx num_edges = csr_col_indices.numel();
  Idx num_heads = grad_sm_first_stage.size(2);
  Idx feat_dim_per_head = grad_sm_first_stage.size(3);
  Idx n_rel_types = grad_sm_first_stage.size(1);
  int nblks = num_nodes;
  int nthrs = num_heads * feat_dim_per_head;
  HET_HGTBackwardGradientSmFirstPartImpl<Idx, DType>
      <<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data,
          grad_sm_first_stage_data, grad_t_neighbour_data, message_data,
          sigmas_data, num_nodes, num_heads, feat_dim_per_head, n_rel_types);

  HET_HGTBackwardGradientAImpl<Idx, DType><<<nblks, nthrs>>>(
      range_data, ids_data, eids_data, typeids_data, grad_a_data,
      grad_t_neighbour_data, message_data, sigmas_data, num_nodes, num_heads,
      feat_dim_per_head, n_rel_types);

  HET_HGTBackwardFusedGradientSmFirstPartGradientAImpl<Idx, DType>
      <<<nblks, nthrs, 0, stream>>>(
          range_data, ids_data, eids_data, typeids_data, grad_a_data,
          grad_sm_first_stage_data, grad_t_neighbour_data, message_data,
          sigmas_data, num_nodes, num_heads, feat_dim_per_head, n_rel_types);
}

// adapted from the BckProp::_full_graph_edge_softmax_ops, the wrapper function
// of HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel
template <typename Idx, typename DType>
void full_graph_EdgeSoftmax_eNorm_to_UnNormalizedAttnScore(
    at::Tensor &incsr_row_ptr, at::Tensor &incsr_col_indices,
    at::Tensor &incsr_eids, at::Tensor &incsr_reltypes,
    at::Tensor &unnormalized_attn_score, at::Tensor &normalized_attn_score,
    at::Tensor &grad_normalized_attn_score, at::Tensor &mu,
    at::Tensor &grad_unnormalized_attn_score, at::Tensor &grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // preparing gdata
  // based on OutputMuAppliedAttnScoreSwitch==2 in
  // HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_edge_softmax_ops
  BackwardNormToUnNormalizedAttnScoreData<Idx, DType> gdata{
      .num_heads = grad_normalized_attn_score.size(
          grad_normalized_attn_score.ndimension() - 1),
      .eids = incsr_eids.data_ptr<Idx>(),
      .grad_normalized_attn_score =
          grad_normalized_attn_score.data_ptr<DType>(),
      .normalized_attn_score = normalized_attn_score.data_ptr<DType>(),
      .grad_mu = grad_mu.data_ptr<DType>(),
      .mu = mu.data_ptr<DType>(),
      .unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>(),
      .grad_unnormalized_attn_score =
          grad_unnormalized_attn_score.data_ptr<DType>()};

  Idx num_relations = mu.numel() / gdata.num_heads;

  // preparing kernel launch configuration
  // NB: Type 1 Schedule:
  // https://github.com/K-Wu/hetero_edgesoftmax/commit/7db47f278d81d10df7af43dabca048c41c5e6382#diff-069c3c2c5a9041df2c9a0b01c9f28044c4d519d86c5ed2f859d0d74282967062L232-R233
  // head -> blockIdx.x * blockDim.x + threadIdx.x;
  // node -> blockIdx.y * blockDim.y + threadIdx.y;
  auto [nblks, nthrs] =
      get_type1_schedule(gdata.num_heads, incsr_row_ptr.numel() - 1);

  HET_EdgeSoftmaxENormToUnNormalizedAttnScoreBackwardKernel<Idx, DType, true,
                                                            false>
      <<<nblks, nthrs, 0, stream>>>(gdata, incsr_row_ptr.data_ptr<Idx>(),
                                    incsr_col_indices.data_ptr<Idx>(),
                                    incsr_reltypes.data_ptr<Idx>(),
                                    incsr_row_ptr.numel() - 1, num_relations);
}

template <typename Idx, typename DType,
          int AttnScoreUseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation_and_edge_softmax(
    at::Tensor &outcsr_row_ptrs, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_reltypes, at::Tensor &outcsr_eids, at::Tensor &message,
    at::Tensor &edgesoftmax_sum_per_node, at::Tensor &unnormalized_attn_score,
    at::Tensor &mu, at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score, at::Tensor &out, at::Tensor &gradout,
    at::Tensor &grad_attn_score, at::Tensor &grad_message,
    at::Tensor &grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // preparing gdata_attn
  // NB: grad_attn_score structure is kept and the unique items from
  // grad_message structure is passed into kernel as individual parameters one
  // by one
  int num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  BackwardHGTAttnScoreData<Idx, DType, AttnScoreUseMuAppliedAttnScoreSwitch>
      gdata_attn{
          .num_heads = num_heads,
          .message_src_xlen =
              message.size(grad_mu.ndimension() - 1) * num_heads,
          .eids = outcsr_eids.data_ptr<Idx>(),
          .grad_attn_score = grad_attn_score.data_ptr<DType>(),
          .message_src = message.data_ptr<DType>(),
          .unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>(),
          .out = out.data_ptr<DType>(),
          .grad_out = gradout.data_ptr<DType>(),
          .grad_mu = grad_mu.data_ptr<DType>(),
          .mu = mu.data_ptr<DType>()};

  assert(gdata_attn.num_heads == message.size(message.ndimension() - 2) &&
         "expecting message.size[-2] to be num_heads but turned out not");

  Idx num_relations = mu.numel() / gdata_attn.num_heads;
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
  int64_t outcsr_num_rows = outcsr_row_ptrs.numel() - 1;
  auto [nblks, nthrs] = get_type2_schedule(
      gdata_attn.num_heads, gdata_attn.message_src_xlen, outcsr_num_rows);

  HET__hgtAttnAndMessageSrcFusedBckKernel<Idx, DType, false, true, false,
                                          AttnScoreUseMuAppliedAttnScoreSwitch>
      <<<nblks, nthrs, 0, stream>>>(
          gdata_attn, grad_message.data_ptr<DType>(),
          outcsr_row_ptrs.data_ptr<Idx>(), outcsr_col_indices.data_ptr<Idx>(),
          outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows,
          /*no need when !CompactAsOfNodeFlag*/ nullptr,
          /*no need when !CompactAsOfNodeFlag*/ nullptr, num_relations);
}

template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor &outcsr_row_ptrs, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_reltypes, at::Tensor &outcsr_eids,
    at::Tensor &edgesoftmax_sum_per_node, at::Tensor &unnormalized_attn_score,
    at::Tensor &mu, at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score, at::Tensor &gradout,
    at::Tensor &grad_message) {
  // using
  // HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel that
  // based on _fusedGatBackwardGradFeatSrc whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // preparing gdata
  int num_heads =
      normalized_attn_score.size(normalized_attn_score.ndimension() - 1);
  BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata{
      .num_heads = num_heads,
      .message_src_xlen =
          grad_message.size(grad_message.ndimension() - 1) * num_heads,
      .eids = outcsr_eids.data_ptr<Idx>(),
      .grad_message_src = grad_message.data_ptr<DType>(),
      .grad_out = gradout.data_ptr<DType>()};
  Idx num_relations = mu.numel() / gdata.num_heads;

  assert(gdata.num_heads == grad_message.size(grad_message.ndimension() - 2) &&
         "assuming num_heads is the same for grad_message but turned out not");

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
  int64_t outcsr_num_rows = outcsr_row_ptrs.numel() - 1;
  auto [nblks, nthrs] = get_type2_schedule(
      gdata.num_heads, gdata.message_src_xlen, outcsr_num_rows);

  HET__hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel<
      Idx, DType, CompactAsOfNodeKind::Disabled, true, false,
      UseMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
      gdata, outcsr_row_ptrs.data_ptr<Idx>(),
      outcsr_col_indices.data_ptr<Idx>(), outcsr_reltypes.data_ptr<Idx>(),
      outcsr_num_rows, {}, num_relations);
}

void full_graph_message_mean_aggregation(
    at::Tensor &outcsr_row_ptrs, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_reltypes, at::Tensor &outcsr_eids,
    at::Tensor &edgesoftmax_sum_per_node, at::Tensor &normalized_attn_score,
    at::Tensor &gradout, at::Tensor &grad_message) {
  at::Tensor dummy_tensor;
  _full_graph_message_mean_aggregation<int64_t, float, 2>(
      outcsr_row_ptrs, outcsr_col_indices, outcsr_reltypes, outcsr_eids,
      edgesoftmax_sum_per_node, dummy_tensor, dummy_tensor, dummy_tensor,
      normalized_attn_score, gradout, grad_message);
}

template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch>
void _full_graph_edge_softmax_ops(
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes, at::Tensor &message,
    at::Tensor &unnormalized_attn_score, at::Tensor &edgesoftmax_sum_per_node,
    at::Tensor &mu_softmax_applied_unnormalized_attn_score,
    at::Tensor &normalized_attn_score, at::Tensor &out, at::Tensor &gradout,
    at::Tensor &mu, at::Tensor &grad_attn_score, at::Tensor &grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  // preparing gdata
  int num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  BackwardHGTAttnScoreData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata{
      .num_heads = num_heads,
      .message_src_xlen = message.size(grad_mu.ndimension() - 1) * num_heads,
      .eids = outcsr_eids.data_ptr<Idx>(),
      .grad_attn_score = grad_attn_score.data_ptr<DType>(),
      .message_src = message.data_ptr<DType>(),
      .unnormalized_attn_score = unnormalized_attn_score.data_ptr<DType>(),
      .out = out.data_ptr<DType>(),
      .grad_out = gradout.data_ptr<DType>(),
      .grad_mu = grad_mu.data_ptr<DType>(),
      .mu = mu.data_ptr<DType>()};

  Idx num_relations = mu.numel() / gdata.num_heads;
  assert(gdata.num_heads == message.size(message.ndimension() - 2) &&
         "expecting message.size[-2] to be num_heads but turned out not");

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
  int64_t outcsr_num_rows = outcsr_row_ptr.numel() - 1;
  auto [nblks, nthrs] = get_type2_schedule(
      gdata.num_heads, gdata.message_src_xlen, outcsr_num_rows);
  // NB: message_src_xlen is the total dimension
  // whereas each head gets message_src_xlen //
  // num_heads number of elements
  HET__hgtEdgeSoftmaxAccumStageOnlyBackwardKernel<
      Idx, DType, CompactAsOfNodeKind::Disabled, true, false,
      OutputMuAppliedAttnScoreSwitch><<<nblks, nthrs, 0, stream>>>(
      gdata, outcsr_row_ptr.data_ptr<Idx>(), outcsr_col_indices.data_ptr<Idx>(),
      outcsr_reltypes.data_ptr<Idx>(), outcsr_num_rows, {}, num_relations);
}

void full_graph_edge_softmax_ops(
    at::Tensor &outcsr_row_ptr, at::Tensor &outcsr_col_indices,
    at::Tensor &outcsr_eids, at::Tensor &outcsr_reltypes, at::Tensor &message,
    at::Tensor &unnormalized_attn_score, at::Tensor &normalized_attn_score,
    at::Tensor &out, at::Tensor &gradout, at::Tensor &mu,
    at::Tensor &grad_attn_score, at::Tensor &grad_mu) {
  at::Tensor dummy_tensor;
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
  _full_graph_edge_softmax_ops<int64_t, float, 2>(
      outcsr_row_ptr, outcsr_col_indices, outcsr_eids, outcsr_reltypes, message,
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
