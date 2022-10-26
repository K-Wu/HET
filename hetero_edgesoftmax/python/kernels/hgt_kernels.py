#!/usr/bin/env python3
import torch


def hgt_full_graph_hetero_message_ops_csr(
    row_ptr, col_idx, eids, reltypes, weight, applied_vlinear_node_features, ret
):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_message_ops_csr(
        row_ptr, col_idx, eids, reltypes, weight, applied_vlinear_node_features, ret
    )


def hgt_full_graph_hetero_message_ops_backward_csr(
    row_ptr,
    col_idx,
    eids,
    reltypes,
    weight,
    applied_vlinear_node_features,
    gradout,
    grad_weight,
    grad_v,
):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward_csr(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        weight,
        applied_vlinear_node_features,
        gradout,
        grad_weight,
        grad_v,
    )


def hgt_full_graph_hetero_attention_ops_csr(
    row_ptr,
    col_idx,
    eids,
    reltypes,
    weight,
    applied_klinear_node_features,
    applied_qlinear_node_features,
    ret,
):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_attention_ops_csr(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        weight,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        ret,
    )


def hgt_full_graph_hetero_attention_ops_backward_csr(
    row_ptr,
    col_idx,
    eids,
    reltypes,
    weight,
    applied_klinear_node_features,
    applied_qlinear_node_features,
    gradout,
    grad_weight,
    grad_k,
    grad_q,
):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_attention_ops_backward_csr(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        weight,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        gradout,
        grad_weight,
        grad_k,
        grad_q,
    )


def hgt_full_graph_message_mean_aggregation_backward_csr(
    transposed_row_ptr,
    transposed_col_idx,
    transposed_eids,
    transposed_reltypes,
    gradout,
    grad_weight,
    grad_attn_score,
):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_message_mean_aggregation_backward_csr(
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        gradout,
        grad_weight,
        grad_attn_score,
    )


def hgt_full_graph_message_mean_aggregation_csr(
    row_ptr,
    col_idx,
    eids,
    reltypes,
    message_per_edge,
    attn_score,
    ret,
):
    return (
        torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_message_mean_aggregation_csr(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            message_per_edge,
            attn_score,
            ret,
        )
    )
