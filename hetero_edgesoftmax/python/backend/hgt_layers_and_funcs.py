#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class HGTFullGraphHeterogeneousMessageOps(th.autograd.Function):
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
        applied_vlinear_node_features,
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
            applied_vlinear_node_features,
        )
        K.hgt_heterogeneous_message_ops_csr(
            row_ptr, col_idx, eids, reltypes, weight, applied_vlinear_node_features, ret
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
            applied_vlinear_node_features,
        ) = ctx.saved_tensors
        print(weight.numel())
        grad_weight = th.zeros_like(weight)
        grad_v = th.zeros_like(applied_vlinear_node_features)
        K.hgt_heterogeneous_message_ops_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            applied_vlinear_node_features,
            grad_weight,
            grad_v,
        )
        return None, None, None, None, None, None, None, None, grad_weight, grad_v, None


class HGTFullGraphHeterogeneousAttentionOps(th.autograd.Function):
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
        K.hgt_heterogeneous_message_ops_csr(
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
        K.hgt_heterogeneous_attention_ops_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            grad_weight,
            grad_k,
            grad_q,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_weight,
            grad_k,
            grad_q,
            None,
        )


def hgt_full_graph_heterogeneous_message_ops_csr(
    graph, weight, applied_vlinear_node_features
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
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return HGTFullGraphHeterogeneousMessageOps.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        weight,
        applied_vlinear_node_features,
        ret,
    )


def hgt_full_graph_heterogeneous_attention_ops_csr(
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
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return HGTFullGraphHeterogeneousAttentionOps.apply(
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
