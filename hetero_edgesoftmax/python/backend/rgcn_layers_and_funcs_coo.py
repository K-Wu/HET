#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class RgcnSecondLayerCOO(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_idx,
        col_idx,
        eids,
        reltypes,
        transposed_row_idx,
        transposed_col_idx,
        transposed_eids,
        tranposed_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            row_idx,
            col_idx,
            eids,
            reltypes,
            transposed_row_idx,
            transposed_col_idx,
            transposed_eids,
            tranposed_reltypes,
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_coo(
            row_idx,
            col_idx,
            eids,
            reltypes,
            x,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            row_idx,
            col_idx,
            eids,
            reltypes,
            transposed_row_idx,
            transposed_col_idx,
            transposed_eids,
            tranposed_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_coo(
            transposed_row_idx,
            transposed_col_idx,
            transposed_eids,
            tranposed_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


def rgcn_layer1_coo(graph, x, weight, norm):
    row_idx = graph["transposed"]["row_idx"]
    col_idx = graph["transposed"]["col_idx"]
    eids = graph["transposed"]["eids"]
    reltypes = graph["transposed"]["rel_types"]
    transposed_row_idx = graph["original"]["row_idx"]
    transposed_col_idx = graph["original"]["col_idx"]
    transposed_eids = graph["original"]["eids"]
    transposed_reltypes = graph["original"]["rel_types"]
    ret = th.zeros(
        (graph.get_num_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnSecondLayerCOO.apply(
        row_idx,
        col_idx,
        eids,
        reltypes,
        transposed_row_idx,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        x,
        weight,
        norm,
        ret,
    )
