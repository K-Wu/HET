#pragma once
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

namespace HET {
namespace TorchExport {
namespace HGT {
namespace FwProp {
void hgt_full_graph_message_mean_aggregation_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& edge_messages,
    at::Tensor& edge_attn_score, at::Tensor& ret) {
  // We need to implement based on (vertex-centric) _gatSumProdZipDivKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]] or
  // (edge-centric) _gatSumProdZipDivKernel_edge_parallel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/RGAT/RGATLayersKernelsSeparateCOO.cu.h]]
}
void hgt_full_graph_hetero_attention_ops_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_klinear_node_features,
    at::Tensor& applied_qlinear_node_features, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_k, at::Tensor& grad_q) {
  // we need to implement a fused kernel based on W*t via RGNN relational_matmul
  // and RGNN inner_product
  assert(0 && "Not implemented yet");
}

void hgt_full_graph_edge_softmax_ops_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& attn_score, at::Tensor& mu,
    at::Tensor& ret) {
  // we need to implement based on _gatExpLeakyReluSumKernel in
  // [[hetero_edgesoftmax/include/DGLHackKernel/GAT/FusedGAT.cu.h]]
  // alternatively there is an existing implementation with tricky API in
  // hetero_edgesoftmax/include/EdgeSoftmax_1/EdgeSoftmaxCSR.h
}
}  // namespace FwProp
namespace BckProp {

void hgt_full_graph_message_mean_aggregation_backward_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& gradout, at::Tensor& grad_message,
    at::Tensor& grad_attn_score) {}

void hgt_full_graph_hetero_attention_ops_backward_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_klinear_node_features,
    at::Tensor& applied_qlinear_node_features, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_k, at::Tensor& grad_q) {
  // we need to implement a fused kernel based on back prop of RGNN
  // inner_product and back prop of W*t via RGNN relational_matmul
  assert(0 && "Not implemented yet");
}

void hgt_full_graph_edge_softmax_ops_backward_wrapper_integratedcsr(
    at::Tensor& transposed_row_ptr, at::Tensor& transposed_col_idx,
    at::Tensor& transposed_eids, at::Tensor& transposed_reltypes,
    at::Tensor& gradout, at::Tensor& grad_attn_score, at::Tensor& grad_mu) {}
}  // namespace BckProp
}  // namespace HGT
}  // namespace TorchExport
}  // namespace HET
