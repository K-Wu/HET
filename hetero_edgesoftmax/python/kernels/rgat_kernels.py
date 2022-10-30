#!/usr/bin/env python3
import torch


def rgat_layer_backward_csr(
    transposed_row_ptr,
    transposed_col_idx,
    transposed_eids,
    transposed_reltypes,
    gradout,
    grad_weight,
):
    return torch.ops.torch_hetero_edgesoftmax.rgat_layer_backward_csr(
        transposed_row_ptr,
        transposed_col_idx,
        transposed_eids,
        transposed_reltypes,
        gradout,
        grad_weight,
    )


def rgat_layer_csr(row_ptr, col_idx, eids, reltypes, weight, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgat_layer_csr(
        row_ptr, col_idx, eids, reltypes, weight, ret
    )
