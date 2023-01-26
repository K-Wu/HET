#!/usr/bin/env python3
import torch


def hgt_full_graph_hetero_attention_ops_coo(
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_coo_eids,
    separate_coo_relptrs,
    applied_klinear_node_features,  # (num_nodes, num_heads, num_hidden)
    applied_qlinear_node_features,  # (num_nodes, num_heads, num_hidden)
    attn_score_weight,  # (num_heads, num_hidden, num_hidden)
    attn_score_inner_product,  # (num_edges, num_heads, num_hidden)
    unnormalized_attn_score,  # (num_edges, num_heads)
):
    # attn_score_inner_product <- applied_klinear_node_features * attn_score_weight
    # unnormalized_attn_score <- attn_score_inner_product * applied_qlinear_node_features
    raise NotImplementedError


def backward_hgt_full_graph_hetero_attention_ops_coo(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_coo_eids,
    separate_coo_relptrs,
    grad_attn_weight,  # (num_heads, num_hidden, num_hidden)
    attn_score_weight_transposed,  # (num_heads, num_hidden, num_hidden)
    applied_klinear_node_features,  # (num_nodes, num_heads, num_hidden)
    applied_qlinear_node_features,  # (num_nodes, num_heads, num_hidden)
    attn_score_inner_product,  # (num_edges, num_heads, num_hidden)
    grad_unnorm_attn_score,  # (num_edges, num_heads)
    grad_k,  # (num_nodes, num_heads, num_hidden)
    grad_q,  # (num_nodes, num_heads, num_hidden)
):
    raise NotImplementedError


def hgt_full_graph_edge_softmax_ops_separate_coo(
    separate_coo_row_indices,
    separate_coo_col_indices,
    separate_coo_eids,
    separate_coo_relptrs,
    unnormalized_attn_score,  # (num_edges, num_heads)
    mu,  # (num_edge_types, num_heads)
    edgesoftmax_sum_per_node,  # (num_nodes, num_heads)
    mu_softmax_applied_unnormalized_attn_score,  # (num_edges, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
):
    raise NotImplementedError


def hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo(
    separate_coo_relptrs,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    inputs,  # (num_nodes, num_heads, num_hidden)
    message_generation_weights,  # (num_heads, num_hidden, num_hidden)
    normalized_attn_score,  # (num_edges, num_heads)
    new_h,  # (num_nodes, num_heads, num_hidden)
):
    raise NotImplementedError


def backward_hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo(
    separate_coo_relptrs,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    inputs,  # (num_nodes, num_heads, num_hidden)
    message_generation_weights_transposed,  # (num_heads, num_hidden, num_hidden)
    normalized_attn_score,  # (num_edges, num_heads)
    new_h,  # (num_nodes, num_heads, num_hidden)
    grad_input,  # (num_nodes, num_heads, num_hidden)
    grad_message_generation_weight,  # (num_heads, num_hidden, num_hidden)
    grad_normalized_attn_score,  # (num_edges, num_heads)
    gradout,  # (num_nodes, num_heads, num_hidden)
):
    raise NotImplementedError


def backward_hgt_full_graph_enorm_to_unnormalized_attn_score_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unnormalized_attn_score,  # (num_edges, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
    grad_normalized_attn_score,  # (num_edges, num_heads)
    mu,  # (num_edge_types, num_heads)
    grad_unnormalized_attn_score,  # (num_edges, num_heads)
    grad_mu,  # (num_edge_types, num_heads)
):
    raise NotImplementedError


def hgt_full_graph_edge_softmax_ops_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unnormalized_attn_score,  # (num_edges, num_heads)
    mu,  # (num_edge_types, num_heads)
    edgesoftmax_sum_per_node,  # (num_nodes, num_heads)
    mu_softmax_applied_unnormalized_attn_score,  # (num_edges, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
):
    raise NotImplementedError


def hgt_full_graph_message_mean_aggregation_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_reltypes,
    incsr_eids,
    message_per_edge,  # (num_edges, num_heads, num_hidden)
    normalized_attn_score,  # (num_edges, num_heads)
    edgesoftmax_sum_per_node,  # (num_nodes, num_heads)
    mu,  # (num_edge_types, num_heads)
    new_h,  # (num_nodes, num_heads, num_hidden)
):
    raise NotImplementedError


def backward_hgt_full_graph_message_mean_aggregation_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    edgesoftmax_sum_per_node,  # (num_nodes, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
    gradout,  # (num_nodes, num_heads, num_hidden)
    grad_message,  # (num_edges, num_heads, num_hidden)
):
    raise NotImplementedError


def backward_hgt_full_graph_edge_softmax_ops_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    message_per_edge,  # (num_edges, num_heads, num_hidden)
    unnormalized_attn_score,  # (num_edges, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
    new_h,  # (num_nodes, num_heads, num_hidden)
    gradout,  # (num_nodes, num_heads, num_hidden)
    mu,  # (num_edge_types, num_heads)
    grad_attn_score,  # (num_edges, num_heads)
    grad_mu,  # (num_edge_types, num_heads)
):
    raise NotImplementedError
