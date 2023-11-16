#!/usr/bin/env python3
import torch
from . import ref_rgnn

from .. import adjacency_manipulation


def hgt_full_graph_hetero_attention_ops_coo(
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_coo_eids,
    separate_coo_relptrs,
    applied_klinear_node_features,  # (num_nodes, num_heads, num_hidden)
    applied_qlinear_node_features,  # (num_nodes, num_heads, num_hidden)
    attn_score_weight,  # (num_edge_types, num_heads, num_hidden, num_hidden)
    attn_score_inner_product,  # (num_edges, num_heads, num_hidden)
    unnormalized_attn_score,  # (num_edges, num_heads)
):
    # attn_score_inner_product <- applied_klinear_node_features * attn_score_weight
    # unnormalized_attn_score <- attn_score_inner_product * applied_qlinear_node_features

    ref_rgnn.rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_relptrs,
        attn_score_weight,
        applied_klinear_node_features,
        attn_score_inner_product,
    )
    attn_score_inner_product[separate_coo_eids] = attn_score_inner_product
    ref_rgnn.rgnn_inner_product_edge_and_node(
        separate_coo_eids,
        separate_coo_row_idx,
        separate_coo_col_idx,
        attn_score_inner_product,
        applied_qlinear_node_features,
        unnormalized_attn_score,
    )


def backward_hgt_full_graph_hetero_attention_ops_coo(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_coo_eids,
    separate_coo_relptrs,
    grad_attn_weight,  # (num_edge_types, num_heads, num_hidden, num_hidden)
    attn_score_weight_transposed,  # (num_heads, num_hidden, num_hidden)
    applied_klinear_node_features,  # (num_nodes, num_heads, num_hidden)
    applied_qlinear_node_features,  # (num_nodes, num_heads, num_hidden)
    attn_score_inner_product,  # (num_edges, num_heads, num_hidden)
    grad_unnorm_attn_score,  # (num_edges, num_heads)
    grad_k,  # (num_nodes, num_heads, num_hidden)
    grad_q,  # (num_nodes, num_heads, num_hidden)
):
    grad_attn_score_inner_product = torch.zeros_like(attn_score_inner_product)
    ref_rgnn.backward_rgnn_inner_product_edge_and_node(
        separate_coo_eids,
        separate_coo_row_idx,
        separate_coo_col_idx,
        attn_score_inner_product,
        applied_qlinear_node_features,
        grad_unnorm_attn_score,
        grad_attn_score_inner_product,
        grad_q,
    )
    ref_rgnn.backward_rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_relptrs,
        attn_score_weight_transposed,
        applied_klinear_node_features,
        grad_attn_score_inner_product[separate_coo_eids],
        grad_attn_weight,
        grad_k,
    )


def _hgt_full_graph_edge_softmax_ops_integrated_coo(
    integrated_coo_row_indices,
    integrated_coo_col_indices,
    integrated_coo_eids,
    integrated_coo_rel_types,
    unnormalized_attn_score,  # (num_edges, num_heads)
    mu,  # (num_edge_types, num_heads)
    edgesoftmax_sum_per_node,  # (num_nodes, num_heads)
    mu_softmax_applied_unnormalized_attn_score,  # (num_edges, num_heads)
    normalized_attn_score,  # (num_edges, num_heads)
):
    mu_per_edge = mu[integrated_coo_rel_types]
    mu_per_edge = mu_per_edge[integrated_coo_eids]
    mu_softmax_applied_unnormalized_attn_score[:] = (
        unnormalized_attn_score * mu_per_edge
    )
    normalized_attn_score[:] = (
        mu_softmax_applied_unnormalized_attn_score
        / edgesoftmax_sum_per_node[integrated_coo_col_indices]
    )


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
    # feat_off_src = edge_id * num_heads + feat_idx;
    # normalized_attn_score[edge_id * num_heads + feat_idx] =
    # unnormalized_attn_score[feat_off_src] * mu /
    # edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) +
    #                                             feat_idx];
    # mu_softmax_applied_unnormalized_attn_score[feat_off_src] = expf(unnormalized_attn_score[feat_off_src] * mu)
    # edgesoftmax_sum_per_node[Idx(dst_vid * num_heads) + feat_idx] += mu_softmax_applied_unnormalized_attn_score[feat_off_src]

    # expand rel_ptrs to (num_edges)
    separate_coo_reltypes = torch.zeros_like(separate_coo_eids)
    for i in range(len(separate_coo_relptrs) - 1):
        separate_coo_reltypes[
            separate_coo_relptrs[i] : separate_coo_relptrs[i + 1]
        ] = i

    _hgt_full_graph_edge_softmax_ops_integrated_coo(
        separate_coo_row_indices,
        separate_coo_col_indices,
        separate_coo_eids,
        separate_coo_reltypes,
        unnormalized_attn_score,
        mu,
        edgesoftmax_sum_per_node,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
    )


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
    # new_h = inputs * message_generation_weights * normalized_attn_score
    feat_per_edge = torch.zeros(
        (normalized_attn_score.shape[0], inputs.shape[1], inputs.shape[2]),
        device=normalized_attn_score.device,
    )  # (num_edges, num_heads, num_hidden)
    ref_rgnn.rgnn_relational_matmul(
        separate_coo_relptrs,
        separate_coo_row_indices,
        separate_coo_eids,
        message_generation_weights,
        inputs,
        feat_per_edge,
        False,
    )
    feat_per_edge = feat_per_edge * normalized_attn_score[:, :, None]
    new_h[:] = torch.index_add(
        new_h, 0, separate_coo_col_indices, feat_per_edge
    )


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
    # grad_input = gradout * message_generation_weights_transposed * normalized_attn_score
    # grad_message_generation_weight = gradout^T * inputs (outer product) * normalized_attn_score
    grad_per_edge = torch.zeros(
        (normalized_attn_score.shape[0], inputs.shape[1], inputs.shape[2]),
        device=normalized_attn_score.device,
    )  # (num_edges, num_heads, num_hidden)
    ref_rgnn.rgnn_relational_matmul(
        separate_coo_relptrs,
        separate_coo_col_indices,
        separate_coo_eids,
        message_generation_weights_transposed,
        gradout,
        grad_per_edge,
        False,
    )
    grad_normalized_attn_score[separate_coo_eids] += torch.sum(
        grad_per_edge[separate_coo_eids] * inputs[separate_coo_row_indices],
        dim=2,
    ).squeeze(2)

    grad_per_edge = grad_per_edge * normalized_attn_score[:, :, None]
    grad_input = torch.index_add(
        grad_input,
        0,
        separate_coo_row_indices,
        grad_per_edge[separate_coo_eids],
    )
    # matmul grad_per_edge (num_edges, num_heads, num_hidden) with inputs[row_indices] (num_edges, num_heads, num_hidden) to get grad_message_generation_weight (num_heads, num_hidden, num_hidden)
    grad_message_generation_weight += torch.bmm(
        (inputs[separate_coo_row_indices]).transpose(0, 2),
        grad_per_edge[separate_coo_eids].transpose(0, 1),
    )


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
    # delta a_j = (-Sum_{i, including i==j} Si*Sj*deltaSi + Sj * deltaSj) * mu_j.
    # delta mu_j = (-Sum_{i, including i==j} Si*Sj*deltaSi + Sj * deltaSj) * a_j.

    # get incsr_row_idx
    incsr_row_idx = adjacency_manipulation._convert_csr_to_coo(
        incsr_row_ptr, incsr_col_idx
    )

    Si_deltaSi_product = grad_normalized_attn_score * normalized_attn_score

    Si_deltaSi_product_sum = torch.zeros(
        size=(incsr_eids.shape[0], unnormalized_attn_score.shape[1]),
        device=unnormalized_attn_score.device,
    )  # (num_nodes, num_heads)
    Si_deltaSi_product_sum = torch.index_add(
        Si_deltaSi_product_sum, 0, incsr_row_idx, Si_deltaSi_product
    )
    grad_unnormalized_attn_score[incsr_eids] = (
        -Si_deltaSi_product_sum[incsr_row_idx]
        * grad_normalized_attn_score[incsr_eids]
        + grad_normalized_attn_score[incsr_eids]
        * normalized_attn_score[incsr_eids]
    ) * mu[incsr_reltypes]

    grad_mu[:] = torch.index_add(
        grad_mu,
        0,
        incsr_reltypes,
        (
            -Si_deltaSi_product_sum[incsr_row_idx]
            * grad_normalized_attn_score[incsr_eids]
            + grad_normalized_attn_score[incsr_eids]
            * normalized_attn_score[incsr_eids]
        )
        * unnormalized_attn_score[incsr_eids],
    )

    # fixme: figure out in other cases if it is incsr or outcsr (done python side, please continue on the C++ code)


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
    # similar to hgt_full_graph_edge_softmax_ops_separate_coo
    # convert row ptr to row idx
    incsr_row_idx = adjacency_manipulation._convert_csr_to_coo(
        incsr_row_ptr, incsr_col_idx
    )

    _hgt_full_graph_edge_softmax_ops_integrated_coo(
        incsr_col_idx,  # in csr is the transpose of the original graph
        incsr_row_idx,
        incsr_eids,
        incsr_reltypes,
        unnormalized_attn_score,
        mu,
        edgesoftmax_sum_per_node,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
    )


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
    # convert row ptr to row idx
    incsr_row_idx = adjacency_manipulation._convert_csr_to_coo(
        incsr_row_ptr, incsr_col_idx
    )

    # new_h += message_per_edge * normalized_attn_score / edgesoftmax_sum_per_node
    new_h[:] = torch.index_add(
        new_h,
        0,
        incsr_row_idx,
        message_per_edge * normalized_attn_score[:, :, None],
    )


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
    # grad_message = gradout * normalized_attn_score / edgesoftmax_sum_per_node
    grad_message[outcsr_eids] = (
        gradout[outcsr_col_idx] * normalized_attn_score[:, :, None]
    )


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
    # FIXME: why there is no such kernel called backward_hgt_full_graph_edge_softmax_ops_separate_coo? Otherwise this should be similar to that

    # grad_mu = grad_out * （message - out) * normalized_attn_score * unnormalized_attn_score
    # grad_attn_score =  grad_out * （message - out) * normalized_attn_score * mu

    grad_mu = torch.index_add(
        grad_mu,
        0,
        outcsr_reltypes,
        gradout[outcsr_col_idx]
        * (message_per_edge[outcsr_eids] - new_h[outcsr_col_idx])
        * normalized_attn_score[outcsr_eids]
        * unnormalized_attn_score[outcsr_eids],
    )
    grad_attn_score[outcsr_eids] = (
        gradout[outcsr_col_idx]
        * (message_per_edge[outcsr_eids] - new_h[outcsr_col_idx])
        * normalized_attn_score[outcsr_eids]
        * mu[outcsr_reltypes]
    )
