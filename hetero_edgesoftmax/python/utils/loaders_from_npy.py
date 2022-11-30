#!/usr/bin/env python3
import numpy as np
import os
from .coo_sorters import sort_coo_by_src_outgoing_edges, sort_coo_by_etype


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


def fetch_fb15k237_dglgraph():
    print("loading fb15k237 from dgl.data")
    from dgl.data import FB15k237Dataset

    dataset = FB15k237Dataset()
    graph = dataset[0]
    return graph


def fetch_fb15k237_raw_data():
    graph = fetch_fb15k237_dglgraph()
    edges_srcs = graph.edges()[0].detach().numpy()
    edges_dsts = graph.edges()[1].detach().numpy()
    edges_etypes = graph.edata["etype"].detach().numpy()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    return edges_srcs, edges_dsts, edges_etypes, edge_referential_eids


def fetch_wikikg2_graph_dict():
    print("loading wikikg2 from ogb.linkproppred")
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-wikikg2")
    graph = dataset[0]
    return graph


def fetch_wikikg2_raw_data():
    graph = fetch_wikikg2_graph_dict()
    edges_srcs = graph["edge_index"][0]
    edges_dsts = graph["edge_index"][1]
    edges_etypes = graph["edge_reltype"].flatten()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    return edges_srcs, edges_dsts, edges_etypes, edge_referential_eids


def fetch_ogbnmag_graph_dict():
    print("loading ogbn-mag from ogb.linkproppred")
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-mag")
    graph = dataset[0]
    return graph


def fetch_ogbnmag_raw_data():
    # NB: we need to reindex nodes as nodes of any type in this data set originally starts with index 0
    # the ordering of the abosolute node indices from 0 to N-1 is author, paper, institution, field_of_study
    graph = fetch_ogbnmag_graph_dict()
    # edges_srcs = graph.edges()[0].detach().numpy()
    # edges_dsts = graph.edges()[0].detach().numpy()
    # edges_etypes = graph.edata['etype'].detach().numpy()
    edge_srcs = graph[0]["edge_index_dict"][
        ("author", "affiliated_with", "institution")
    ][0]
    edge_dsts = (
        graph[0]["edge_index_dict"][("author", "affiliated_with", "institution")][1]
        + graph[0]["num_nodes_dict"]["author"]
        + graph[0]["num_nodes_dict"]["paper"]
    )
    edge_types = [0] * len(edge_srcs)

    edge_srcs2 = graph[0]["edge_index_dict"][("author", "writes", "paper")][0]
    edge_dsts2 = (
        graph[0]["edge_index_dict"][("author", "writes", "paper")][1]
        + graph[0]["num_nodes_dict"]["author"]
    )
    edge_types2 = [1] * len(edge_srcs2)

    edge_srcs3 = (
        graph[0]["edge_index_dict"][("paper", "cites", "paper")][0]
        + graph[0]["num_nodes_dict"]["author"]
    )
    edge_dsts3 = (
        graph[0]["edge_index_dict"][("paper", "cites", "paper")][1]
        + graph[0]["num_nodes_dict"]["author"]
    )
    edge_types3 = [2] * len(edge_srcs3)

    edge_srcs4 = (
        graph[0]["edge_index_dict"][("paper", "has_topic", "field_of_study")][0]
        + graph[0]["num_nodes_dict"]["author"]
    )
    edge_dsts4 = (
        graph[0]["edge_index_dict"][("paper", "has_topic", "field_of_study")][1]
        + graph[0]["num_nodes_dict"]["author"]
        + graph[0]["num_nodes_dict"]["paper"]
        + graph[0]["num_nodes_dict"]["institution"]
    )
    edge_types4 = [3] * len(edge_srcs4)
    return (
        np.concatenate([edge_srcs, edge_srcs2, edge_srcs3, edge_srcs4]),
        np.concatenate([edge_dsts, edge_dsts2, edge_dsts3, edge_dsts4]),
        np.concatenate([edge_types, edge_types2, edge_types3, edge_types4]),
        np.arange(
            len(edge_srcs) + len(edge_srcs2) + len(edge_srcs3) + len(edge_srcs4),
            dtype=np.int64,
        ),
    )


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
        return fetch_fb15k237_raw_data()
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
        return fetch_wikikg2_raw_data()
    else:
        return generic_load_data(
            os.path.join(
                dataset_path_prefix,
                (transposed_prefix + "wikikg2" + ".coo" + sorted_suffix),
            )
        )


def get_ogbnmag(sorted, sorted_by_srcs, transposed, infidel_sort_flag: bool = False):
    if sorted_by_srcs and (not sorted):
        raise ValueError("sorted_by_srcs is only valid when sorted is True")
    if infidel_sort_flag:
        print("Warning: you are loading infidel sort data, see readme.md for details")

    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids = fetch_ogbnmag_raw_data()
    if transposed:
        edge_srcs, edge_dsts = edge_dsts, edge_srcs
    if sorted:
        if sorted_by_srcs:
            (
                edge_srcs,
                edge_dsts,
                edge_etypes,
                edge_referential_eids,
            ) = sort_coo_by_src_outgoing_edges(
                edge_srcs,
                edge_dsts,
                edge_etypes,
                edge_referential_eids,
                torch_flag=False,
                infidel_sort_flag=infidel_sort_flag,
            )
        else:
            (
                edge_srcs,
                edge_dsts,
                edge_etypes,
                edge_referential_eids,
            ) = sort_coo_by_etype(
                edge_srcs,
                edge_dsts,
                edge_etypes,
                edge_referential_eids,
                torch_flag=False,
                infidel_sort_flag=infidel_sort_flag,
            )
    return edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
