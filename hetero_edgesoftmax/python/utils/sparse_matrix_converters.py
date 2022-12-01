#!/usr/bin/env python3
import numpy as np
import torch as th


@th.no_grad()
def coo2csr(
    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids, torch_flag: bool = False
):
    if torch_flag:
        curr_namespace = th
    else:
        curr_namespace = np
    # Sort by srcs
    sorted_indices = curr_namespace.argsort(edge_srcs)
    edge_srcs = edge_srcs[sorted_indices]
    edge_dsts = edge_dsts[sorted_indices]
    edge_etypes = edge_etypes[sorted_indices]
    edge_referential_eids = edge_referential_eids[sorted_indices]

    # compress rows
    row_offsets = curr_namespace.zeros(edge_srcs.max() + 2, dtype=curr_namespace.int64)
    row_offsets[1:] = curr_namespace.bincount(edge_srcs)
    row_offsets = curr_namespace.cumsum(
        row_offsets, 0
    )  # the second argument is named axis in numpy and dim in torch

    return (
        row_offsets,
        edge_dsts,
        edge_etypes,
        edge_referential_eids,
    )  # the returned variables are row_ptr, col_idx, rel_types, eids, respectively


@th.no_grad()
def csr2coo(row_ptr, col_idx, rel_types, eids):
    # we don't need to provide csr2coo for torch because coo should be the natural product of applying loading utility onto a torch Tensor
    # And torch does not provide funcitons like torch.repeat counterpart to numpy.repeat anyway.

    # expand rows
    edge_srcs = np.repeat(np.arange(row_ptr.size - 1), np.diff(row_ptr))
    edge_dsts = col_idx
    edge_etypes = rel_types
    edge_referential_eids = eids

    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
