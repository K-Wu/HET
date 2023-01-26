#!/usr/bin/env python3
import torch
from torch.autograd.function import FunctionCtx
from dgl.backend.pytorch import *

# sparse.SEGMENTMM


def rgnn_relational_matmul_ac_gather_scatter_list_identical(
    separate_coo_relptrs,
    separate_coo_eids,
    weights,  # (num_heads, in_dim, feat_dim)
    inputs,  # (num edges, num_heads, in_dim)
    ret,  # (num edges, num_heads, feat_dim)
    input_num_head_one_flag,
):
    raise NotImplementedError


def backward_rgnn_relational_matmul_ac_gather_scatter_list_identical(
    separate_coo_relptrs,
    separate_coo_eids,
    weights_transposed,  # (num_heads, feat_dim, in_dim)
    inputs,  # (num edges, num_heads, in_dim)
    gradout,  # (num edges, num_heads, feat_dim)
    grad_input,  # (num edges, num_heads, in_dim)
    grad_weight,  # (num_heads, in_dim, feat_dim)
    input_num_head_one_flag,
):
    raise NotImplementedError


def rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    weights,  # (num_heads, in_dim, feat_dim)
    inputs,  # (node num, num_heads, in_dim)
    ret,  # (num edges, num_heads, feat_dim)
    input_num_head_one_flag,
):
    raise NotImplementedError


def backward_rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    weights_transposed,  # (num_heads, feat_dim, in_dim)
    inputs,  # (node num, num_heads, in_dim)
    gradout,  # (num edges, num_heads, feat_dim)
    grad_input,  # (node num, num_heads, in_dim)
    grad_weight,  # (num_heads, in_dim, feat_dim)
    input_num_head_one_flag,
):
    raise NotImplementedError


def rgnn_relational_matmul_no_scatter_gather_list(
    ntype_offset_ptrs,
    weights,  # (num_heads, in_dim, feat_dim)
    inputs,  # (node num, num_heads, in_dim)
    ret,  # (node num, num_heads, feat_dim)
):
    raise NotImplementedError


def backward_rgnn_relational_matmul_no_scatter_gather_list(
    ntype_offset_ptrs,
    weights_transposed,
    inputs,  # (node num, num_heads, in_dim)
    gradout,  # (node num, num_heads, feat_dim)
    grad_weight,  # (num_heads, in_dim, feat_dim)
    grad_input,  # (node num, num_heads, in_dim)
):
    raise NotImplementedError


def rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    weight,  # (num_heads, in_dim, feat_dim)
    node_feat,  # (node num, num_heads, in_dim)
    ret,  # (unique (node, rel) num, num_heads, feat_dim)
    input_num_head_one_flag,
):
    raise NotImplementedError


def backward_rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    weight_transposed,
    node_feat,
    gradout,
    grad_node_feat,
    grad_weight,
    input_num_head_one_flag,
):
    raise NotImplementedError


def rgnn_inner_product_node_compact_and_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_node_compact_data,  # (unique (node, rel) num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    ret,  # (edge_num, num_heads)
):
    raise NotImplementedError


def backward_rgnn_inner_product_node_compact_and_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_node_compact_data,  # (unique (node, rel) num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    gradout,  # (edge_num, num_heads)
    grad_left_node_compact_data,
    grad_right_node_vectors,
):
    raise NotImplementedError


def rgnn_inner_product_edge_and_node(
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_edge_data,  # (edge_num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    ret,  # (edge_num, num_heads)
):
    raise NotImplementedError


def backward_rgnn_inner_product_edge_and_node(
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_edge_data,  # (edge_num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    gradout,  # (edge_num, num_heads)
    grad_left_edge_data,
    grad_right_node_vectors,
):

    raise NotImplementedError
