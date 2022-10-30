#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class RgatLayerCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        weight,
        ret,
    ):
        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            weight,
        )
        K.rgat_layer_csr(
            incsr_row_ptr, incsr_col_idx, incsr_eids, incsr_reltypes, weight, ret
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            weight,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weight)
        K.rgat_layer_backward_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            gradout,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_weight, None


def rgat_layer_csr(graph, weight):
    # NB: notice that in rgcn, in-adjacency list is used and therefore, we input the transposed adj list to forward propagation, and the original to backward propagation.
    incsr_row_ptr = graph["transposed"]["row_ptr"]
    incsr_col_idx = graph["transposed"]["col_idx"]
    incsr_eids = graph["transposed"]["eids"]
    incsr_reltypes = graph["transposed"]["rel_types"]
    outcsr_row_ptr = graph["original"]["row_ptr"]
    outcsr_col_idx = graph["original"]["col_idx"]
    outcsr_eids = graph["original"]["eids"]
    outcsr_reltypes = graph["original"]["rel_types"]
    ret = th.zeros(
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgatLayerCSR.apply(
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        ret,
    )
