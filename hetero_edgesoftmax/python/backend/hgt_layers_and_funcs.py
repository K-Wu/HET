#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class HGTFullGraphHeteroAttentionOps(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        weight,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        ret,
    ):

        ctx.save_for_backward(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            weight,
            applied_klinear_node_features,
            applied_qlinear_node_features,
        )
        K.hgt_full_graph_hetero_attention_ops_csr(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            weight,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            weight,
            applied_klinear_node_features,
            applied_qlinear_node_features,
        ) = ctx.saved_tensors
        print(weight.numel())
        grad_weight = th.zeros_like(weight)
        grad_k = th.zeros_like(applied_klinear_node_features)
        grad_q = th.zeros_like(applied_qlinear_node_features)
        K.hgt_full_graph_hetero_attention_ops_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            weight,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            gradout,
            grad_weight,
            grad_k,
            grad_q,
        )
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return None, None, None, None, None, None, None, None, grad_weight, grad_k, grad_q, None
        # fmt: on


class HGTFullGraphMessageMeanAggregationCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        message_per_edge,
        attn_score,
        ret,
    ):

        ctx.save_for_backward(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            message_per_edge,
            attn_score,
        )
        K.hgt_full_graph_message_mean_aggregation_csr(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            message_per_edge,
            attn_score,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            message_per_edge,
            attn_score,
        ) = ctx.saved_tensors
        grad_message = th.zeros_like(message_per_edge)
        grad_attn_score = th.zeros_like(attn_score)
        K.hgt_full_graph_message_mean_aggregation_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            grad_message,
            grad_attn_score,
        )
        # fmt: off
        return None, None, None, None, None, None, None, None, grad_message, grad_attn_score, None
        # fmt: on


def hgt_full_graph_hetero_attention_ops_csr(
    graph, weight, applied_klinear_node_features, applied_qlinear_node_features
):
    row_ptr = graph["original"]["row_ptr"]
    col_idx = graph["original"]["col_idx"]
    eids = graph["original"]["eids"]
    reltypes = graph["original"]["rel_types"]
    transposed_row_ptr = graph["transposed"]["row_ptr"]
    transposed_col_idx = graph["transposed"]["col_idx"]
    transposed_eids = graph["transposed"]["eids"]
    transposed_reltypes = graph["transposed"]["rel_types"]
    ret = th.zeros(
        (
            graph["original"]["row_ptr"].numel() - 1,
            weight.size(2),
        ),  # weight size (self.num_relations, n_heads, self.d_k, self.d_k)
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return HGTFullGraphHeteroAttentionOps.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        weight,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        ret,
    )


def hgt_full_graph_message_mean_aggregation_csr(graph, message_per_edge, attn_score):
    row_ptr = graph["original"]["row_ptr"]
    col_idx = graph["original"]["col_idx"]
    eids = graph["original"]["eids"]
    reltypes = graph["original"]["rel_types"]
    transposed_row_ptr = graph["transposed"]["row_ptr"]
    transposed_col_idx = graph["transposed"]["col_idx"]
    transposed_eids = graph["transposed"]["eids"]
    transposed_reltypes = graph["transposed"]["rel_types"]
    ret = th.zeros(
        graph["original"]["num_nodes"],
        message_per_edge.size(1),
        message_per_edge.size(2),
        dtype=message_per_edge.dtype,
        device=message_per_edge.device,
        requires_grad=True,
    )
    return HGTFullGraphMessageMeanAggregationCSR.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        message_per_edge,
        attn_score,
        ret,
    )


class HGTFullGraphEdgeSoftmaxOpsCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        attn_score,
        mu,
        ret,
    ):

        ctx.save_for_backward(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            attn_score,
            mu,
        )
        K.hgt_full_graph_edge_softmax_ops_csr(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            attn_score,
            mu,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            attn_score,
            mu,
        ) = ctx.saved_tensors
        grad_attn_score = th.zeros_like(attn_score)
        grad_mu = th.zeros_like(mu)
        K.hgt_full_graph_edge_softmax_ops_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            grad_attn_score,
            grad_mu,
        )
        # fmt: off
        return None, None, None, None, None, None, None, None, grad_attn_score, grad_mu, None
        # fmt: on


def hgt_full_graph_edge_softmax_ops_csr(graph, attn_score, mu):
    row_ptr = graph["original"]["row_ptr"]
    col_idx = graph["original"]["col_idx"]
    eids = graph["original"]["eids"]
    reltypes = graph["original"]["rel_types"]
    transposed_row_ptr = graph["transposed"]["row_ptr"]
    transposed_col_idx = graph["transposed"]["col_idx"]
    transposed_eids = graph["transposed"]["eids"]
    transposed_reltypes = graph["transposed"]["rel_types"]
    ret = th.zeros_like(attn_score)
    # scale_factor, i.e., sqrt_dk equals math.sqrt(out_dim // n_heads)
    return HGTFullGraphEdgeSoftmaxOpsCSR.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        attn_score,
        mu,
        ret,
    )
