#!/usr/bin/env python3
import torch


def transpose_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.transpose_csr(*args)


def convert_integrated_csr_to_separate_csr(
    *args,
    # row_ptr, col_idx, rel_types, eids
):
    return torch.ops.torch_hetero_edgesoftmax.convert_integrated_csr_to_separate_csr(
        *args
    )


def convert_integrated_csr_to_separate_coo(
    *args,
    # row_ptr, col_idx, rel_types, eids
):
    return torch.ops.torch_hetero_edgesoftmax.convert_integrated_csr_to_separate_coo(
        *args
    )


def convert_integrated_coo_to_separate_csr(
    *args,
    # row_idx, col_idx, rel_types, eids, num_rows, num_rels
):
    return torch.ops.torch_hetero_edgesoftmax.convert_integrated_coo_to_separate_csr(
        *args
    )


def convert_integrated_coo_to_separate_coo(
    *args,
    # row_idx, col_idx, rel_types, eids, num_rows, num_rels
):
    return torch.ops.torch_hetero_edgesoftmax.convert_integrated_coo_to_separate_coo(
        *args
    )
