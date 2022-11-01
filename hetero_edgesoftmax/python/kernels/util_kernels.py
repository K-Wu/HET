#!/usr/bin/env python3
import torch


def transpose_csr(row_ptr, col_idx, eids, reltypes):
    return torch.ops.torch_hetero_edgesoftmax.transpose_csr(
        row_ptr, col_idx, eids, reltypes
    )
