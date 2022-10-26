#!/usr/bin/env python3
import numpy as np
import os
import torch as th
from . import sparse_matrix_converters


class MyDGLGraph:
    def __init__(self):
        self.graph_data = dict()

    def __setitem__(self, key, value):
        self.graph_data[key] = value

    def __getitem__(self, key):
        return self.graph_data[key]

    def to(self, device):
        for key in self.graph_data:
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][second_key].to(
                    device
                )

    def cuda(self):
        for key in self.graph_data:
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][
                    second_key
                ].cuda()


def convert_mydgl_graph_csr_to_coo(g):
    # we haven't implemented csr2coo for tensors so we need to convert to numpy first
    row_ptr = g["original"]["row_ptr"].numpy()
    col_idx = g["original"]["col_idx"].numpy()
    rel_types = g["original"]["rel_types"].numpy()
    eids = g["original"]["eids"].numpy()
    (
        edge_srcs,
        edge_dsts,
        edge_etypes,
        edge_referential_eids,
    ) = sparse_matrix_converters.csr2coo(row_ptr, col_idx, rel_types, eids)
    return create_mydgl_graph_coo_numpy(
        edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
    )


def convert_mydgl_graph_coo_to_csr(g):
    (
        edge_srcs,
        edge_dsts,
        edge_etypes,
        edge_referential_eids,
    ) = sparse_matrix_converters.coo2csr(
        g["original"]["row_ptr"],
        g["original"]["col_idx"],
        g["original"]["rel_types"],
        g["original"]["eids"],
        torch_flag=True,
    )
    return create_mydgl_graph_csr_torch(
        edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
    )


def create_mydgl_graph_csr_torch(row_ptr, col_idx, rel_types, eids):
    g = MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_ptr"] = row_ptr
    g["original"]["col_idx"] = col_idx
    g["original"]["rel_types"] = rel_types
    g["original"]["eids"] = eids
    return g


def create_mydgl_graph_csr_numpy(row_ptr, col_idx, rel_types, eids):
    row_ptr = th.from_numpy(row_ptr).long()
    col_idx = th.from_numpy(col_idx).long()
    rel_types = th.from_numpy(rel_types).long()
    eids = th.from_numpy(eids).long()
    return create_mydgl_graph_csr_torch(row_ptr, col_idx, rel_types, eids)


def create_mydgl_graph_coo_from_dgl_graph(g):
    total_edge_srcs = th.zeros(g.number_of_edges(), dtype=th.int64)
    total_edge_dsts = th.zeros(g.number_of_edges(), dtype=th.int64)
    total_edge_etypes = th.zeros(g.number_of_edges(), dtype=th.int64)
    total_edge_referential_eids = th.arange(g.number_of_edges(), dtype=th.int64)
    etype_offsets = np.zeros(len(g.etypes) + 1, dtype=np.int64)
    for etype_idx, etype in enumerate(g.canonical_etypes):
        last_etype_offsets = etype_offsets[etype_idx - 1] if etype_idx > 0 else 0
        etype_offsets[etype_idx] = g.number_of_edges(etype=etype) + last_etype_offsets
        edge_srcs, edge_dsts = g.edges(etype=etype)  # both are int64 Torch.Tensor
        print(
            etype,
            "edge_srcs \in [",
            min(edge_srcs),
            max(edge_srcs),
            "], edge_dests \in [",
            min(edge_dsts),
            max(edge_dsts),
            "]",
        )
        # add to total
        total_edge_srcs[last_etype_offsets : etype_offsets[etype_idx]] = edge_srcs
        total_edge_dsts[last_etype_offsets : etype_offsets[etype_idx]] = edge_dsts
        total_edge_etypes[last_etype_offsets : etype_offsets[etype_idx]] = etype_idx
    return create_mydgl_graph_coo_torch(
        total_edge_srcs, total_edge_dsts, total_edge_etypes, total_edge_referential_eids
    )


def create_mydgl_graph_coo_torch(
    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
):
    g = MyDGLGraph()
    g["original"] = dict()
    g["original"]["srcs"] = edge_srcs
    g["original"]["dsts"] = edge_dsts
    g["original"]["etypes"] = edge_etypes
    g["original"]["eids"] = edge_referential_eids
    return g


def create_mydgl_graph_coo_numpy(
    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
):
    edge_srcs = th.from_numpy(edge_srcs).long()
    edge_dsts = th.from_numpy(edge_dsts).long()
    edge_etypes = th.from_numpy(edge_etypes).long()
    edge_referential_eids = th.from_numpy(edge_referential_eids).long()
    return create_mydgl_graph_coo_torch(
        edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
    )


def generic_load_data(dataset_path_and_name_prefix):
    # load the data
    # example of dataset_path_and_name_prefix: in load_fb15k237 and load_wikikg2, it should be os.path.join(dataset_path_prefix,(transposed_prefix+dataset_name+".coo"+sorted_suffix))
    edge_srcs = np.load(dataset_path_and_name_prefix + ".srcs.npy", allow_pickle=True)
    edge_dsts = np.load(dataset_path_and_name_prefix + ".dsts.npy", allow_pickle=True)
    edge_etypes = np.load(
        dataset_path_and_name_prefix + ".etypes.npy", allow_pickle=True
    )
    edge_referential_eids = np.load(
        dataset_path_and_name_prefix + ".referential_eids.npy", allow_pickle=True
    )
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids


def get_fb15k237_data():
    print("loading fb15k237 from dgl.data")
    from dgl.data import FB15k237Dataset

    dataset = FB15k237Dataset()
    graph = dataset[0]
    edges_srcs = graph.edges()[0].detach().numpy()
    edges_dsts = graph.edges()[1].detach().numpy()
    edges_etypes = graph.edata["etype"].detach().numpy()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    return edges_srcs, edges_dsts, edges_etypes, edge_referential_eids


def get_wikikg2_data():
    print("loading wikikg2 from ogb.linkproppred")
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-wikikg2")
    graph = dataset[0]
    edges_srcs = graph["edge_index"][0]
    edges_dsts = graph["edge_index"][1]
    edges_etypes = graph["edge_reltype"].flatten()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    return edges_srcs, edges_dsts, edges_etypes, edge_referential_eids


def load_fb15k237(
    dataset_path_prefix, sorted, sorted_by_srcs, transposed, infidel_sort_flag=True
):
    if sorted_by_srcs and (not sorted):
        raise ValueError("sorted_by_srcs is only valid when sorted is True")
    transposed_prefix = "transposed." if transposed else ""
    if infidel_sort_flag:
        print("Warning: you are loading infidel sort data, see readme.md for details")
        sorted_suffix = ".infidel_sorted" if sorted else ""
    else:
        sorted_suffix = ".sorted" if sorted else ""
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"
    elif sorted:
        sorted_suffix += ".by_etype_freq"

    if not sorted:
        return get_fb15k237_data()
    else:  # sorted
        return generic_load_data(
            os.path.join(
                dataset_path_prefix,
                (transposed_prefix + "fb15k237" + ".coo" + sorted_suffix),
            )
        )


def load_wikikg2(
    dataset_path_prefix, sorted, sorted_by_srcs, transposed, infidel_sort_flag=False
):
    if sorted_by_srcs and (not sorted):
        raise ValueError("sorted_by_srcs is only valid when sorted is True")
    transposed_prefix = "transposed." if transposed else ""
    if infidel_sort_flag:
        print("Warning: you are loading infidel sort data, see readme.md for details")
        sorted_suffix = ".infidel_sorted" if sorted else ""
    else:
        sorted_suffix = ".sorted" if sorted else ""
    if sorted and sorted_by_srcs:
        sorted_suffix += ".by_srcs_outgoing_freq"
    if not sorted:
        return get_wikikg2_data()
    else:
        return generic_load_data(
            os.path.join(
                dataset_path_prefix,
                (transposed_prefix + "wikikg2" + ".coo" + sorted_suffix),
            )
        )
