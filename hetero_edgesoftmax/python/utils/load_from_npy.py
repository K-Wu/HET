#!/usr/bin/env python3
import numpy as np
import os

def pyutils_load_fb15k237(dataset_path_prefix, sorted, sorted_by_srcs,transposed):
    transposed_prefix = "transposed." if transposed else ''
    sorted_suffix = ".sorted" if sorted else ''
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"
    edge_srcs = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "fb15k237.coo"+sorted_suffix+".srcs.npy")), allow_pickle=True)
    edge_dsts = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "fb15k237.coo"+sorted_suffix+".dsts.npy")), allow_pickle=True)
    edge_etypes = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "fb15k237.coo"+sorted_suffix+".etypes.npy")), allow_pickle=True)
    edge_referential_eids = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "fb15k237.coo"+sorted_suffix+".referential_eids.npy")), allow_pickle=True)
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids

def pyutils_load_wikikg2(dataset_path_prefix, sorted, sorted_by_srcs,transposed):
    transposed_prefix = "transposed." if transposed else ''
    sorted_suffix = ".sorted" if sorted else ''
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"
    
    edge_srcs = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "ogbn-wikikg2.coo"+sorted_suffix+".srcs.npy")), allow_pickle=True)
    edge_dsts = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "ogbn-wikikg2.coo"+sorted_suffix+".dsts.npy")), allow_pickle=True)
    edge_etypes = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "ogbn-wikikg2.coo"+sorted_suffix+".etypes.npy")), allow_pickle=True)
    edge_referential_eids = np.load(os.path.join(dataset_path_prefix, (transposed_prefix + "ogbn-wikikg2.coo"+sorted_suffix+".referential_eids.npy")), allow_pickle=True)
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids


