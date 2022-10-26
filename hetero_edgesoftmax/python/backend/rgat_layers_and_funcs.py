#!/usr/bin/env python3
import torch as th

from .. import kernels as K


class RgatLayerCSR(th.autograd.Function):
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
        )
        K.rgat_layer_csr(row_ptr, col_idx, eids, reltypes, weight, ret)
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
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weight)
        K.rgat_layer_backward_csr(
            transposed_row_ptr,
            transposed_col_idx,
            transposed_eids,
            transposed_reltypes,
            gradout,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_weight, None


def rgat_layer_csr(graph, weight):
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
    return RgatLayerCSR.apply(
        row_ptr,
        col_idx,
        eids,
        reltypes,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        ret,
    )
