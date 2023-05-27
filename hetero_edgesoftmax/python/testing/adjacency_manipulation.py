#!/usr/bin/env python3
import torch


def _convert_csr_to_coo(
    original_row_ptrs: torch.Tensor,
    original_col_idxes: torch.Tensor,
):
    row_idxes = torch.zeros(original_col_idxes.size(0), dtype=torch.long)
    for i in range(original_row_ptrs.size(0) - 1):
        row_idxes[original_row_ptrs[i] : original_row_ptrs[i + 1]] = i
    return row_idxes


def convert_csr_to_coo(
    original_row_ptrs: torch.Tensor,
    original_col_idxes: torch.Tensor,
    original_eids: torch.Tensor,
    original_reltypes: torch.Tensor,
):

    row_idxes = _convert_csr_to_coo(original_row_ptrs, original_col_idxes)
    return row_idxes, original_col_idxes, original_eids, original_reltypes


def convert_coo_to_csr(
    original_row_idxes, original_col_idxes, original_eids, original_reltypes
):
    # sort by row_idxes
    sorted_row_idxes, sorted_idxes = torch.sort(original_row_idxes)
    sorted_col_idxes = original_col_idxes[sorted_idxes]
    sorted_eids = original_eids[sorted_idxes]
    sorted_reltypes = original_reltypes[sorted_idxes]
    # get row_ptrs
    row_ptrs = torch.zeros(original_row_idxes.max() + 2, dtype=torch.long)
    for i in range(sorted_row_idxes.size(0)):
        row_ptrs[sorted_row_idxes[i] + 1] += 1
    row_ptrs = torch.cumsum(row_ptrs, dim=0)
    return row_ptrs, sorted_col_idxes, sorted_eids, sorted_reltypes


def is_coo_equal(
    row_idxes1: torch.Tensor,
    col_idxes1: torch.Tensor,
    eids1: torch.Tensor,
    reltypes1: torch.Tensor,
    row_idxes2: torch.Tensor,
    col_idxes2: torch.Tensor,
    eids2: torch.Tensor,
    reltypes2: torch.Tensor,
):
    # we need to consider cases where there are permutated edges

    # first, sort the edges by eids
    sorted_eids1, sorted_idxes1 = torch.sort(eids1)
    sorted_eids2, sorted_idxes2 = torch.sort(eids2)
    assert torch.equal(sorted_eids1, sorted_eids2)
    row_idxes1 = row_idxes1[sorted_idxes1]
    col_idxes1 = col_idxes1[sorted_idxes1]
    reltypes1 = reltypes1[sorted_idxes1]
    row_idxes2 = row_idxes2[sorted_idxes2]
    col_idxes2 = col_idxes2[sorted_idxes2]
    reltypes2 = reltypes2[sorted_idxes2]
    return (
        torch.equal(row_idxes1, row_idxes2)
        and torch.equal(col_idxes1, col_idxes2)
        and torch.equal(eids1, eids2)
        and torch.equal(reltypes1, reltypes2)
    )


def transpose_csr(
    original_row_ptrs: torch.Tensor,
    original_col_idxes: torch.Tensor,
    original_eids: torch.Tensor,
    original_reltypes: torch.Tensor,
):
    # the reference implementation of transpose_csr
    # (
    #     transposed_row_ptrs,
    #     transposed_col_idxes,
    #     transposed_eids,
    #     transposed_reltypes,
    # ) = K.transpose_csr(
    #     original_row_ptrs, original_col_idxes, original_eids, original_reltypes
    # )

    # convert to coo, transpose and convert to csr
    row_idxes, col_idxes, eids, reltypes = convert_csr_to_coo(
        original_row_ptrs, original_col_idxes, original_eids, original_reltypes
    )
    transposed_row_idxes = col_idxes
    transposed_col_idxes = row_idxes
    transposed_eids = eids
    transposed_reltypes = reltypes
    (
        transposed_row_ptrs,
        transposed_col_idxes,
        transposed_eids,
        transposed_reltypes,
    ) = convert_coo_to_csr(
        transposed_row_idxes, transposed_col_idxes, transposed_eids, transposed_reltypes
    )
    return (
        transposed_row_ptrs,
        transposed_col_idxes,
        transposed_eids,
        transposed_reltypes,
    )


def convert_integrated_coo_to_separate_coo(row_idx, col_idx, rel_types, eids):
    # reference implementation of convert_integrated_coo_to_separate_coo
    # (
    #     separate_coo_rel_ptr,
    #     separate_coo_row_idx,
    #     separate_coo_col_idx,
    #     separate_coo_eids,
    # ) = K.convert_integrated_coo_to_separate_coo(
    #     self.graph_data[original_or_transposed]["row_indices"],
    #     self.graph_data[original_or_transposed]["col_indices"],
    #     self.graph_data[original_or_transposed]["rel_types"],
    #     self.graph_data[original_or_transposed]["eids"],
    # )

    # sort by rel_types, and then calculate rel_ptr like in csr
    sorted_rel_types, sorted_idxes = torch.sort(rel_types)
    sorted_row_idx = row_idx[sorted_idxes]
    sorted_col_idx = col_idx[sorted_idxes]
    sorted_eids = eids[sorted_idxes]
    rel_ptr = torch.zeros(sorted_rel_types.max() + 2, dtype=torch.long)
    for i in range(sorted_rel_types.size(0)):
        rel_ptr[sorted_rel_types[i] + 1] += 1
    rel_ptr = torch.cumsum(rel_ptr, dim=0)
    return rel_ptr, sorted_row_idx, sorted_col_idx, sorted_eids


def calc_inverse_mapping_of_eids(separate_coo_eids):
    inverse_separate_coo_eids = torch.zeros(
        (separate_coo_eids.shape[0],),
        dtype=torch.int64,
        device=separate_coo_eids.device,
    )
    inverse_separate_coo_eids[separate_coo_eids] = torch.arange(
        separate_coo_eids.shape[0], device=separate_coo_eids.device
    )
    return inverse_separate_coo_eids
