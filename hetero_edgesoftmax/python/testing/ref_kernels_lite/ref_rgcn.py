#!/usr/bin/env python3


def rgcn_layer1_separate_coo(
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_idx,
    separate_coo_col_idx,
    x,  # (node_num, in_dim)
    weight,  # (in_dim, feat_dim)
    norm,  # (edge_num)
    ret,  # (edge_num, feat_dim)
):
    raise NotImplementedError


def backward_rgcn_layer1_separate_coo(
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_idx,
    separate_coo_col_idx,
    x,  # (node_num, in_dim)
    weight_transposed,  # (in_dim, feat_dim)
    norm,  # (edge_num)
    grad_norm,  # (edge_num)
    grad_x,  # (node_num, in_dim)
    gradout,  # (edge_num, feat_dim)
    grad_weight,  # (in_dim, feat_dim)
):
    raise NotImplementedError


def rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (node_num, in_dim)
    enorm,  # (edge_num)
    ret,  # (node_num, feat_dim)
):
    raise NotImplementedError


def backward_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (node_num, in_dim)
    enorm,  # (edge_num)
    ret,  # (node_num, feat_dim)
    gradout,  # (node_num, feat_dim)
    grad_feat_src,  # (node_num, in_dim)
):
    raise NotImplementedError
