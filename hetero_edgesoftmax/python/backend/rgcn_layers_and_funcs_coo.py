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
        in_row_idx,
        in_col_idx,
        in_eids,
        in_reltypes,
        out_row_idx,
        out_col_idx,
        out_eids,
        out_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            in_row_idx,
            in_col_idx,
            in_eids,
            in_reltypes,
            out_row_idx,
            out_col_idx,
            out_eids,
            out_reltypes,
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_coo(
            in_row_idx,
            in_col_idx,
            in_eids,
            in_reltypes,
            x,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            in_row_idx,
            in_col_idx,
            in_eids,
            in_reltypes,
            out_row_idx,
            out_col_idx,
            out_eids,
            out_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_coo(
            out_row_idx,
            out_col_idx,
            out_eids,
            out_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


def rgcn_layer1_coo(graph, x, weight, norm):
    in_row_idx = graph["transposed"]["row_idx"]
    in_col_idx = graph["transposed"]["col_idx"]
    in_eids = graph["transposed"]["eids"]
    in_reltypes = graph["transposed"]["rel_types"]
    out_row_idx = graph["original"]["row_idx"]
    out_col_idx = graph["original"]["col_idx"]
    out_eids = graph["original"]["eids"]
    transposed_reltypes = graph["original"]["rel_types"]
    ret = th.zeros(
        (graph.get_num_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnSecondLayerCOO.apply(
        in_row_idx,
        in_col_idx,
        in_eids,
        in_reltypes,
        out_row_idx,
        out_col_idx,
        out_eids,
        transposed_reltypes,
        x,
        weight,
        norm,
        ret,
    )
