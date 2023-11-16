#!/usr/bin/env python3
import torch
from . import ref_rgnn


def rgcn_layer1_separate_coo(
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_idx,
    separate_coo_col_idx,
    x,  # (node_num, in_dim)
    weight,  # (num_edge_types, in_dim, feat_dim)
    norm,  # (edge_num)
    ret,  # (node_num, feat_dim)
):
    feat_per_edge = torch.index_select(x, 0, separate_coo_row_idx)
    weight.unsqueeze(1)
    ref_rgnn.rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_rel_ptr, weight, feat_per_edge, feat_per_edge
    )
    feat_per_edge = feat_per_edge * norm.unsqueeze(1)
    ret[:] = torch.index_add(ret, 0, separate_coo_col_idx, feat_per_edge)


def backward_rgcn_layer1_separate_coo(
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_idx,
    separate_coo_col_idx,
    x,  # (node_num, in_dim)
    weight_transposed,  # (num_edge_types, in_dim, feat_dim)
    norm,  # (edge_num)
    grad_norm,  # (edge_num)
    grad_x,  # (node_num, in_dim)
    gradout,  # (node_num, feat_dim)
    grad_weight,  # (in_dim, feat_dim)
):
    grad_feat_per_edge = gradout[separate_coo_col_idx]
    grad_feat_per_edge = grad_feat_per_edge * norm.unsqueeze(1)
    grad_weight[:] = torch.matmul(
        x[separate_coo_row_idx].t(), grad_feat_per_edge
    )
    weight_transposed.unsqueeze(1)
    ref_rgnn.rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_rel_ptr,
        weight_transposed,
        grad_feat_per_edge,
        grad_feat_per_edge,
    )

    feat_per_edge_for_grad_norm = torch.index_select(
        x, 0, separate_coo_row_idx
    )
    grad_norm[:] = torch.sum(
        feat_per_edge_for_grad_norm * grad_feat_per_edge, dim=1
    )

    grad_x[:] = torch.index_add(
        grad_x, 0, separate_coo_row_idx, grad_feat_per_edge
    )


def rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (node_num, feat_dim)
    enorm,  # (edge_num)
    ret,  # (node_num, feat_dim)
):
    # compact version of rgcn_layer1_separate_coo
    feat_per_edge = torch.index_select(feat_src, 0, separate_coo_row_idx)
    feat_per_edge = feat_per_edge * enorm.unsqueeze(1)
    ret[:] = torch.index_add(ret, 0, separate_coo_col_idx, feat_per_edge)


def backward_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (node_num, feat_dim)
    enorm,  # (edge_num)
    ret,  # (node_num, feat_dim)
    gradout,  # (node_num, feat_dim)
    grad_feat_src,  # (node_num, feat_dim)
):
    # compact version of backward_rgcn_layer1_separate_coo

    grad_feat_per_edge = gradout[separate_coo_col_idx]
    grad_feat_per_edge = grad_feat_per_edge * enorm.unsqueeze(1)

    grad_feat_src[:] = torch.index_add(
        grad_feat_src, 0, separate_coo_row_idx, grad_feat_per_edge
    )
