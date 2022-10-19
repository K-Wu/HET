import numpy as np


def pyutils_coo2csr(edge_srcs, edge_dsts, edge_etypes, edge_referential_eids):
    # Sort by srcs
    sorted_indices = np.argsort(edge_srcs)
    edge_srcs = edge_srcs[sorted_indices]
    edge_dsts = edge_dsts[sorted_indices]
    edge_etypes = edge_etypes[sorted_indices]
    edge_referential_eids = edge_referential_eids[sorted_indices]

    # compress rows
    row_offsets = np.zeros(edge_srcs.max() + 2, dtype=np.int64)
    row_offsets[1:] = np.bincount(edge_srcs)
    row_offsets = np.cumsum(row_offsets)


    return row_offsets, edge_dsts, edge_etypes, edge_referential_eids  # the returned variables are row_ptr, col_idx, rel_types, eids, respectively

def pyutils_csr2coo(row_ptr, col_idx, rel_types, eids):
    # expand rows
    edge_srcs = np.repeat(np.arange(row_ptr.size - 1), np.diff(row_ptr))
    edge_dsts = col_idx
    edge_etypes = rel_types
    edge_referential_eids = eids


    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids