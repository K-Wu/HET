#!/usr/bin/env python3
import numpy as np
import os
import torch as th

def create_mydgl_graph_csr(row_ptr, col_idx, rel_types, eids):
    row_ptr = th.from_numpy(row_ptr).long()
    col_idx = th.from_numpy(col_idx).long()
    rel_types = th.from_numpy(rel_types).long()
    eids = th.from_numpy(eids).long()
    g = dict()
    g["original"] = dict()
    g["original"]["row_ptr"] = row_ptr
    g["original"]["col_idx"] = col_idx
    g["original"]["rel_types"] = rel_types
    g["original"]["eids"] = eids
    return g


def create_mydgl_graph_coo(edge_srcs, edge_dsts, edge_etypes, edge_referential_eids):
    edge_srcs = th.from_numpy(edge_srcs).long()
    edge_dsts = th.from_numpy(edge_dsts).long()
    edge_etypes = th.from_numpy(edge_etypes).long()
    edge_referential_eids = th.from_numpy(edge_referential_eids).long()

    g = dict()
    g["original"] = dict()
    g["original"]["srcs"] = edge_srcs
    g["original"]["dsts"] = edge_dsts
    g["original"]["etypes"] = edge_etypes
    g["original"]["eids"] = edge_referential_eids


def pyutils_load_fb15k237(dataset_path_prefix, sorted, sorted_by_srcs, transposed):
    transposed_prefix = "transposed." if transposed else ""
    sorted_suffix = ".sorted" if sorted else ""
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"
    edge_srcs = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "fb15k237.coo" + sorted_suffix + ".srcs.npy"),
        ),
        allow_pickle=True,
    )
    edge_dsts = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "fb15k237.coo" + sorted_suffix + ".dsts.npy"),
        ),
        allow_pickle=True,
    )
    edge_etypes = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "fb15k237.coo" + sorted_suffix + ".etypes.npy"),
        ),
        allow_pickle=True,
    )
    edge_referential_eids = np.load(
        os.path.join(
            dataset_path_prefix,
            (
                transposed_prefix
                + "fb15k237.coo"
                + sorted_suffix
                + ".referential_eids.npy"
            ),
        ),
        allow_pickle=True,
    )
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids


def pyutils_load_wikikg2(dataset_path_prefix, sorted, sorted_by_srcs, transposed):
    transposed_prefix = "transposed." if transposed else ""
    sorted_suffix = ".sorted" if sorted else ""
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"

    edge_srcs = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "ogbn-wikikg2.coo" + sorted_suffix + ".srcs.npy"),
        ),
        allow_pickle=True,
    )
    edge_dsts = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "ogbn-wikikg2.coo" + sorted_suffix + ".dsts.npy"),
        ),
        allow_pickle=True,
    )
    edge_etypes = np.load(
        os.path.join(
            dataset_path_prefix,
            (transposed_prefix + "ogbn-wikikg2.coo" + sorted_suffix + ".etypes.npy"),
        ),
        allow_pickle=True,
    )
    edge_referential_eids = np.load(
        os.path.join(
            dataset_path_prefix,
            (
                transposed_prefix
                + "ogbn-wikikg2.coo"
                + sorted_suffix
                + ".referential_eids.npy"
            ),
        ),
        allow_pickle=True,
    )
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
