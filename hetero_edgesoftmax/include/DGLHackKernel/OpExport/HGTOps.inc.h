#pragma once
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include "DGLHackKernel/HGT/HGTBackwardKernels.cu.h"
#include "DGLHackKernel/HGT/HGTExperimental.cu.h"
#include "DGLHackKernel/HGT/HGTForwardKernels.cu.h"
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

// TODO: add torch tensor version of HGTForwardImpl from
// [[hetero_edgesoftmax/include/DGLHackKernel/OpPrototyping/HGTProtoOps.h]]

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {
namespace IntegratedCSR {
template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor& incsr_rowptr, at::Tensor& incsr_col_idx,
    at::Tensor& incsr_reltypes, at::Tensor& incsr_eids,
    at::Tensor& edge_messages, at::Tensor& edge_attn_score, at::Tensor& ret) {
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
  Idx incsr_num_rows = incsr_rowptr.numel() - 1;
  if (num_heads <= 1) {
    std::cout << "Warning: num_heads <= 1 in "
                 "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
                 "edge_softmax_ops"
              << std::endl;
  }

  // configure kernel launch parameters
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
}

void full_graph_message_mean_aggregation(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& gradout, at::Tensor& grad_message,
    at::Tensor& grad_attn_score) {
  _full_graph_message_mean_aggregation<int64_t, float, 1>(
      csr_rowptr, csr_col_idx, csr_reltypes, csr_eids, gradout, grad_message,
      grad_attn_score);
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
template <typename Idx, typename DType, int OutputMuAppliedAttnScoreSwitch>
void _full_graph_edge_softmax_ops(at::Tensor& incsr_rowptr,
                                  at::Tensor& incsr_col_idx,
                                  at::Tensor& incsr_eids,
                                  at::Tensor& incsr_reltypes,
                                  at::Tensor& attn_score, at::Tensor& mu,
                                  at::Tensor& ret) {
  // We need to implement based on _gatExpLeakyReluSumKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]].
  // There is an existing implementation with tricky API in
  // hetero_edgesoftmax/include/EdgeSoftmax_1/EdgeSoftmaxCSR.h
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // Configure kernel parameters structure
  HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads = attn_score.size(attn_score.ndimension() - 1);
  if (num_heads <= 1) {
    std::cout << "Warning: num_heads <= 1 in "
                 "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
                 "edge_softmax_ops"
              << std::endl;
  }
  // Configure kernel launch parameters. From _gatExpLeakyReluSumKernel
  // configurations in
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl
  int nthrs_x = 32;
  int nthrs_y = 1;
  int nblks_x = (num_heads + nthrs_x - 1) / (nthrs_x);
  int64_t incsr_num_rows = incsr_rowptr.numel() - 1;
  int nblks_y = std::min(incsr_num_rows, MAX_NBLKS);
  const dim3 nblks(nblks_x, nblks_y);
  const dim3 nthrs(nthrs_x, nthrs_y);
}

void full_graph_edge_softmax_ops(at::Tensor& row_ptr, at::Tensor& col_idx,
                                 at::Tensor& eids, at::Tensor& reltypes,
                                 at::Tensor& attn_score, at::Tensor& mu,
                                 at::Tensor& ret) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
}

// this function only calculates edge softmax sum at each destination node.
void full_graph_edge_softmax_only_accumu_stage_ops(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& attn_score, at::Tensor& mu,
    at::Tensor& ret) {
  // using _hgtEdgeSoftmaxAccumStageOnlyKernel based on
  // _gatExpLeakyReluSumKernel whose driver code is
  // HET::TorchExport::RGCN::FwProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]
  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that does only the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
}

}  // namespace IntegratedCSR
}  // namespace FwProp
namespace BckProp {
namespace IntegratedCSR {

// from HGTBackPropGradientSMAFusion in
// [[hetero_edgesoftmax/include/DGLHackKernel/OpPrototyping/HGTProtoOps.h]]
template </*int XPU, */ typename Idx, typename DType>
void HGTBackPropGradientSMAFusion(
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

template <typename Idx, typename DType, int UseMuAppliedAttnScoreSwitch>
void _full_graph_message_mean_aggregation(
    at::Tensor& outcsr_rowptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_reltypes, at::Tensor& outcsr_eids, at::Tensor& gradout,
    at::Tensor& grad_message, at::Tensor& grad_attn_score) {
  // using _hgtMessageAccumBasedOnOriAttnScoreAndEdgeSoftmaxSumBackwardKernel
  // that based on _fusedGatBackwardGradFeatSrc whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;

  // preparing gdata
  BackwardHGTMessageData<Idx, DType, UseMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  if (num_heads <= 1) {
    std::cout << "Warning: num_heads <= 1 in "
                 "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
                 "edge_softmax_ops"
              << std::endl;
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
}

void full_graph_message_mean_aggregation(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& gradout, at::Tensor& grad_message,
    at::Tensor& grad_attn_score) {
  _full_graph_message_mean_aggregation<int64_t, float, 1>(
      csr_rowptr, csr_col_idx, csr_reltypes, csr_eids, gradout, grad_message,
      grad_attn_score);
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
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& gradout,
    at::Tensor& grad_attn_score, at::Tensor& grad_mu) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const Idx MAX_NBLKS = 65535;
  const Idx MAX_NTHRS = 1024;
  // preparing gdata
  HgtEdgeSoftmaxAccumData<Idx, DType, OutputMuAppliedAttnScoreSwitch> gdata;
  Idx num_heads = grad_attn_score.size(grad_attn_score.ndimension() - 1);
  if (num_heads <= 1) {
    std::cout << "Warning: num_heads <= 1 in "
                 "HET::TorchExport::HGT::BckProp::IntegratedCSR::_full_graph_"
                 "edge_softmax_ops"
              << std::endl;
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
}

void full_graph_edge_softmax_ops(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& gradout,
    at::Tensor& grad_attn_score, at::Tensor& grad_mu) {
  // calling the partial specialized version of _full_graph_edge_softmax_ops
  // that does both stages, i.e., MuAppliedAttnScoreSwitch == 2
}

void full_graph_edge_softmax_only_accumu_stage_ops(
    at::Tensor& outcsr_row_ptr, at::Tensor& outcsr_col_idx,
    at::Tensor& outcsr_eids, at::Tensor& outcsr_reltypes, at::Tensor& gradout,
    at::Tensor& grad_attn_score, at::Tensor& grad_mu) {
  // using _hgtEdgeSoftmaxAccumStageOnlyBackwardKernel based on
  // _fusedGatBackwardGradElEr whose driver code is
  // HET::TorchExport::RGCN::BckProp::IntegratedCSR::_FusedKernelImpl in
  // [[hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h]]

  // TODO: call the partial specialized version of _full_graph_edge_softmax_ops
  // that only do the first stage, i.e., MuAppliedAttnScoreSwitch == 0|1
}

}  // namespace IntegratedCSR
}  // namespace BckProp
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET
