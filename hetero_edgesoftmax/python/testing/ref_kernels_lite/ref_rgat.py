#!/usr/bin/env python3
import torch


def relational_fused_gat_kernel_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num_edges, num_heads)
    er,  # (num_edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num_edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    raise NotImplementedError


def backward_relational_fused_gat_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num_edges, num_heads)
    er,  # (num_edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num_edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num_edges, num_heads)
    grad_er,  # (num_edges, num_heads)
    slope,
):
    raise NotImplementedError


def relational_fused_gat_kernel_compact_as_of_node_separate_coo_dual_unique_node_list(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr_row,
    separate_unique_node_idx_rel_ptr_col,
    separate_unique_node_idx_node_idx_row,
    separate_unique_node_idx_node_idx_col,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    raise NotImplementedError


def backward_relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr_row,
    separate_unique_node_idx_rel_ptr_col,
    separate_unique_node_idx_node_idx_row,
    separate_unique_node_idx_node_idx_col,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num (etype, unique node idx), num_heads)
    grad_er,  # (num (etype, unique node idx), num_heads)
    slope,
):
    raise NotImplementedError


def relational_fused_gat_kernel_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    raise NotImplementedError


def backward_relational_fused_gat_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num (etype, unique node idx), num_heads)
    grad_er,  # (num (etype, unique node idx), num_heads)
    slope,
):
    raise NotImplementedError


def relational_fused_gat_kernel_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num edges, num_heads)
    er,  # (num edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    raise NotImplementedError


def backward_relational_fused_gat_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num edges, num_heads)
    er,  # (num edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num edges, num_heads)
    grad_er,  # (num edges, num_heads)
    slope,
):
    raise NotImplementedError


def rgat_relational_fused_gat_compact_as_of_node_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    el_compact,  # (num (etype, unique node idx), num_heads)
    er_compact,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    raise NotImplementedError


def backward_rgat_relational_fused_gat_compact_as_of_node_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    el_compact,  # (num (etype, unique node idx), num_heads)
    er_compact,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    grad_el_compact,  # (num (etype, unique node idx), num_heads)
    grad_er_compact,  # (num (etype, unique node idx), num_heads)
    slope,
):
    raise NotImplementedError
