#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class RgcnFirstLayerCSR(th.autograd.Function):
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
        norm,
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
            norm,
        )
        K.rgcn_layer0_csr(row_ptr, col_idx, eids, reltypes, weight, norm, ret)
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
            norm,
        ) = ctx.saved_tensors
        print(weight.numel())
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer0_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            norm,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_weight, None, None


def rgcn_layer0_csr(graph, weight, norm):
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
    return RgcnFirstLayerCSR.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        weight,
        norm,
        ret,
    )


class RgcnSecondLayerCSR(th.autograd.Function):
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
        tranposed_reltypes,
        x,
        weight,
        norm,
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
            tranposed_reltypes,
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_csr(
            row_ptr,
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
            row_ptr,
            col_idx,
            eids,
            reltypes,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            tranposed_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_csr(
            transposed_row_ptr,
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


def rgcn_layer1_csr(graph, x, weight, norm):
    row_ptr = graph["original"]["row_ptr"]
    col_idx = graph["original"]["col_idx"]
    eids = graph["original"]["eids"]
    reltypes = graph["original"]["rel_types"]
    transposed_row_ptr = graph["transposed"]["row_ptr"]
    transposed_col_idx = graph["transposed"]["col_idx"]
    transposed_eids = graph["transposed"]["eids"]
    transposed_reltypes = graph["transposed"]["rel_types"]
    ret = th.zeros(
        (row_ptr.numel() - 1, weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnSecondLayerCSR.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        x,
        weight,
        norm,
        ret,
    )
