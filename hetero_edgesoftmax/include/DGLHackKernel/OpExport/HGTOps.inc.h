#pragma once
#include "DGLHackKernel/OpExport/HGTPrepToAndFromTensors.h"
#include "EdgeSoftmax_1/EdgeSoftmaxCSR.h"

void hgt_full_graph_message_mean_aggregation_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& edge_messages,
    at::Tensor& edge_attn_score, at::Tensor& ret) {}

void hgt_full_graph_message_mean_aggregation_backward_wrapper_integratedcsr(
    at::Tensor& csr_rowptr, at::Tensor& csr_col_idx, at::Tensor& csr_reltypes,
    at::Tensor& csr_eids, at::Tensor& gradout, at::Tensor& grad_message,
    at::Tensor& grad_attn_score) {}

void hgt_full_graph_hetero_attention_ops_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_klinear_node_features,
    at::Tensor& applied_qlinear_node_features, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_k, at::Tensor& grad_q) {}

void hgt_full_graph_hetero_attention_ops_backward_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_klinear_node_features,
    at::Tensor& applied_qlinear_node_features, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_k, at::Tensor& grad_q) {}

void hgt_full_graph_hetero_message_ops_backward_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_vlinear_node_features, at::Tensor& gradout,
    at::Tensor& grad_weight, at::Tensor& grad_v) {}

void hgt_full_graph_hetero_message_ops_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& weight,
    at::Tensor& applied_klinear_node_features,
    at::Tensor& applied_qlinear_node_features, at::Tensor& ret) {}

void hgt_full_graph_edge_softmax_ops_wrapper_integratedcsr(
    at::Tensor& row_ptr, at::Tensor& col_idx, at::Tensor& eids,
    at::Tensor& reltypes, at::Tensor& attn_score, at::Tensor& mu,
    at::Tensor& ret) {}

void hgt_full_graph_edge_softmax_ops_backward_wrapper_integratedcsr(
    at::Tensor& transposed_row_ptr, at::Tensor& transposed_col_idx,
    at::Tensor& transposed_eids, at::Tensor& transposed_reltypes,
    at::Tensor& gradout, at::Tensor& grad_attn_score, at::Tensor& grad_mu) {}