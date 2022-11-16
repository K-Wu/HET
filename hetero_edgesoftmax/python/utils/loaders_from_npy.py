#!/usr/bin/env python3
import numpy as np
import os
import torch as th
from . import sparse_matrix_converters
from . import mydgl_graph
from .coo_sorters import sort_coo_by_src_outgoing_edges, sort_coo_by_etype


def RGNN_get_mydgl_graph(
    dataset, sort_by_src_flag, sort_by_etype_flag, reindex_eid_flag, sparse_format
):
    # TODO: add args for dataset, and refactor these following lines into dedicated load data function
    # load graph data
    # data_rowptr, data_colidx, data_reltypes, data_eids
    # transposed_data_rowptr, transposed_data_colidx, transposed_data_reltypes, transposed_data_eids,

    dataset_sort_flag = sort_by_src_flag or sort_by_etype_flag

    if dataset == "fb15k":
        print("WARNING - loading fb15k. Currently we only support a few dataset.")
        (edge_srcs, edge_dsts, edge_etypes, edge_referential_eids,) = load_fb15k237(
            "data/MyHybData",
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=False,
            infidel_sort_flag=False,
        )
        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = load_fb15k237(
            "data/MyHybData",
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=True,
            infidel_sort_flag=False,
        )
        dglgraph = fetch_fb15k237_dglgraph()
    elif dataset == "wikikg2":
        print("WARNING - loading wikikg2. Currently we only support a few dataset.")
        (edge_srcs, edge_dsts, edge_etypes, edge_referential_eids,) = load_wikikg2(
            "data/MyWikiKG2",
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=False,
            infidel_sort_flag=False,
        )
        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = load_wikikg2(
            "data/MyWikiKG2",
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=True,
            infidel_sort_flag=False,
        )
        dglgraph = fetch_wikikg2_dglgraph()
    elif dataset == "ogbnmag":
        print("WARNING - loading mag. Currently we only support a few dataset.")
        (edge_srcs, edge_dsts, edge_etypes, edge_referential_eids,) = get_ogbnmag(
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=False,
            infidel_sort_flag=False,
        )
        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = get_ogbnmag(
            dataset_sort_flag,
            sort_by_src_flag,
            transposed=True,
            infidel_sort_flag=False,
        )
        dglgraph = fetch_ogbnmag_dglgraph()
    else:
        print("ERROR! now only support fb15k and wikikg2. Loading it now")
        exit(1)
    if reindex_eid_flag:
        edge_new_eids = np.arange(edge_referential_eids.shape[0])
        edge_referential_to_new_eids_mapping = dict(
            zip(edge_referential_eids, edge_new_eids)
        )
        transposed_edge_new_eids = np.array(
            list(
                map(
                    edge_referential_to_new_eids_mapping.__getitem__,
                    transposed_edge_referential_eids,
                )
            )
        ).astype(np.int64)
        print("transposed_edge_new_eids shape", transposed_edge_new_eids.shape)
        edge_referential_eids = edge_new_eids
        transposed_edge_referential_eids = transposed_edge_new_eids

    if sparse_format == "coo":
        g = create_mydgl_graph_coo_with_transpose_numpy(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        )
    elif sparse_format == "csr":
        # coo to csr conversion
        (
            data_rowptr,
            data_colidx,
            data_reltypes,
            data_eids,
        ) = sparse_matrix_converters.coo2csr(
            edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
        )
        (
            transposed_data_rowptr,
            transposed_data_colidx,
            transposed_data_reltypes,
            transposed_data_eids,
        ) = sparse_matrix_converters.coo2csr(
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        )
        # create graph
        g = create_mydgl_graph_csr_with_transpose_numpy(
            data_rowptr,
            data_colidx,
            data_reltypes,
            data_eids,
            transposed_data_rowptr,
            transposed_data_colidx,
            transposed_data_reltypes,
            transposed_data_eids,
        )
    else:
        raise NotImplementedError("sparse format not supported")
    g.import_metadata_from_dgl_heterograph(dglgraph)
    return g


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
    if "transposed" not in g:
        return create_mydgl_graph_coo_numpy(
            edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
        )
    else:
        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = sparse_matrix_converters.csr2coo(
            g["transposed"]["row_ptr"].numpy(),
            g["transposed"]["col_idx"].numpy(),
            g["transposed"]["rel_types"].numpy(),
            g["transposed"]["eids"].numpy(),
        )
        return create_mydgl_graph_coo_with_transpose_numpy(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
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
    if "transposed" not in g:
        return create_mydgl_graph_csr_torch(
            edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
        )
    else:
        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = sparse_matrix_converters.coo2csr(
            g["transposed"]["row_ptr"],
            g["transposed"]["col_idx"],
            g["transposed"]["rel_types"],
            g["transposed"]["eids"],
            torch_flag=True,
        )
        return create_mydgl_graph_csr_with_transpose_torch(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        )


def create_mydgl_graph_csr_with_transpose_torch(
    row_ptr,
    col_idx,
    rel_types,
    eids,
    transposed_row_ptr,
    transposed_col_idx,
    transposed_rel_types,
    transposed_eids,
):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_ptr"] = row_ptr
    g["original"]["col_idx"] = col_idx
    g["original"]["rel_types"] = rel_types
    g["original"]["eids"] = eids
    g["transposed"] = dict()
    g["transposed"]["row_ptr"] = transposed_row_ptr
    g["transposed"]["col_idx"] = transposed_col_idx
    g["transposed"]["rel_types"] = transposed_rel_types
    g["transposed"]["eids"] = transposed_eids
    return g


def create_mydgl_graph_csr_torch(row_ptr, col_idx, rel_types, eids):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_ptr"] = row_ptr
    g["original"]["col_idx"] = col_idx
    g["original"]["rel_types"] = rel_types
    g["original"]["eids"] = eids
    return g


def create_mydgl_graph_csr_with_transpose_numpy(
    row_ptr,
    col_idx,
    rel_types,
    eids,
    transposed_row_ptr,
    transposed_col_idx,
    transposed_rel_types,
    transposed_eids,
):
    row_ptr = th.from_numpy(row_ptr).long()
    col_idx = th.from_numpy(col_idx).long()
    rel_types = th.from_numpy(rel_types).long()
    eids = th.from_numpy(eids).long()
    transposed_row_ptr = th.from_numpy(transposed_row_ptr).long()
    transposed_col_idx = th.from_numpy(transposed_col_idx).long()
    transposed_rel_types = th.from_numpy(transposed_rel_types).long()
    transposed_eids = th.from_numpy(transposed_eids).long()
    return create_mydgl_graph_csr_with_transpose_torch(
        row_ptr,
        col_idx,
        rel_types,
        eids,
        transposed_row_ptr,
        transposed_col_idx,
        transposed_rel_types,
        transposed_eids,
    )


def create_mydgl_graph_csr_numpy(row_ptr, col_idx, rel_types, eids):
    row_ptr = th.from_numpy(row_ptr).long()
    col_idx = th.from_numpy(col_idx).long()
    rel_types = th.from_numpy(rel_types).long()
    eids = th.from_numpy(eids).long()
    return create_mydgl_graph_csr_torch(row_ptr, col_idx, rel_types, eids)


@th.no_grad()
def create_mydgl_graph_coo_from_dgl_graph(g):
    # total_edge_srcs = th.zeros(g.number_of_edges(), dtype=th.int64)
    # total_edge_dsts = th.zeros(g.number_of_edges(), dtype=th.int64)
    # total_edge_etypes = th.zeros(g.number_of_edges(), dtype=th.int64)
    # total_edge_referential_eids = th.arange(g.number_of_edges(), dtype=th.int64)
    etype_offsets = np.zeros(len(g.etypes) + 1, dtype=np.int64)

    # calculate the offsets for each node type. See the following NB for more details.
    ntype_offsets = np.zeros(len(g.ntypes) + 1, dtype=np.int64)
    ntype_offsets[0] = 0
    ntype_id_map = dict()
    for idx, ntype in enumerate(g.ntypes):
        ntype_offsets[idx + 1] = ntype_offsets[idx] + g.number_of_nodes(ntype)
        ntype_id_map[ntype] = idx
    edge_srcs_list = []
    edge_dsts_list = []
    edge_etypes_list = []
    # edge_referential_eids_list = []
    for etype_idx, etype in enumerate(g.canonical_etypes):
        last_etype_offsets = etype_offsets[etype_idx - 1] if etype_idx > 0 else 0
        etype_offsets[etype_idx] = g.number_of_edges(etype=etype) + last_etype_offsets
        print("getting view for etype", etype)
        edge_srcs, edge_dsts = g.edges(etype=etype)  # both are int64 Torch.Tensor
        print("got view for etype", etype)
        # NB: we here add offsets to edge_srcs and edge_dsts because indices restart from 0 in every new node type
        edge_srcs = edge_srcs + ntype_offsets[ntype_id_map[etype[0]]]
        edge_dsts = edge_dsts + ntype_offsets[ntype_id_map[etype[2]]]
        print("added offsets for etype", etype)
        # print(
        #     etype,
        #     "edge_srcs \in [",
        #     min(edge_srcs),
        #     max(edge_srcs),
        #     "], edge_dests \in [",
        #     min(edge_dsts),
        #     max(edge_dsts),
        #     "]",
        # )
        # add to total
        # total_edge_srcs[last_etype_offsets : etype_offsets[etype_idx]] = edge_srcs
        # total_edge_dsts[last_etype_offsets : etype_offsets[etype_idx]] = edge_dsts
        # total_edge_etypes[last_etype_offsets : etype_offsets[etype_idx]] = etype_idx
        edge_srcs_list.append(edge_srcs)
        edge_dsts_list.append(edge_dsts)
        edge_etypes_list.append(th.full_like(edge_srcs, etype_idx))
        print("added to total for etype", etype)
    total_edge_srcs = th.cat(edge_srcs_list)
    total_edge_dsts = th.cat(edge_dsts_list)
    total_edge_etypes = th.cat(edge_etypes_list)
    total_edge_referential_eids = th.arange(g.number_of_edges(), dtype=th.int64)
    mydgl_graph = create_mydgl_graph_coo_torch(
        total_edge_srcs, total_edge_dsts, total_edge_etypes, total_edge_referential_eids
    )
    mydgl_graph.import_metadata_from_dgl_heterograph(g)
    return mydgl_graph


def create_mydgl_graph_coo_with_transpose_torch(
    edge_srcs,
    edge_dsts,
    edge_etypes,
    edge_eids,
    transposed_edge_srcs,
    transposed_edge_dsts,
    transposed_edge_etypes,
    transposed_edge_eids,
):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_idx"] = edge_srcs
    g["original"]["col_idx"] = edge_dsts
    g["original"]["rel_types"] = edge_etypes
    g["original"]["eids"] = edge_eids
    g["transposed"] = dict()
    g["transposed"]["row_idx"] = transposed_edge_srcs
    g["transposed"]["col_idx"] = transposed_edge_dsts
    g["transposed"]["rel_types"] = transposed_edge_etypes
    g["transposed"]["eids"] = transposed_edge_eids
    return g


def create_mydgl_graph_coo_torch(
    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_idx"] = edge_srcs
    g["original"]["col_idx"] = edge_dsts
    g["original"]["rel_types"] = edge_etypes
    g["original"]["eids"] = edge_referential_eids
    return g


def create_mydgl_graph_coo_with_transpose_numpy(
    edge_srcs,
    edge_dsts,
    edge_etypes,
    edge_eids,
    transposed_edge_srcs,
    transposed_edge_dsts,
    transposed_edge_etypes,
    transposed_edge_eids,
):
    edge_srcs = th.from_numpy(edge_srcs).long()
    edge_dsts = th.from_numpy(edge_dsts).long()
    edge_etypes = th.from_numpy(edge_etypes).long()
    edge_eids = th.from_numpy(edge_eids).long()
    transposed_edge_srcs = th.from_numpy(transposed_edge_srcs).long()
    transposed_edge_dsts = th.from_numpy(transposed_edge_dsts).long()
    transposed_edge_etypes = th.from_numpy(transposed_edge_etypes).long()
    transposed_edge_eids = th.from_numpy(transposed_edge_eids).long()
    return create_mydgl_graph_coo_with_transpose_torch(
        edge_srcs,
        edge_dsts,
        edge_etypes,
        edge_eids,
        transposed_edge_srcs,
        transposed_edge_dsts,
        transposed_edge_etypes,
        transposed_edge_eids,
    )


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


def fetch_wikikg2_dglgraph():
    print("loading wikikg2 from ogb.linkproppred")
    from ogb.linkproppred import LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-wikikg2")
    graph = dataset[0]
    return graph


def fetch_wikikg2_raw_data():
    graph = fetch_wikikg2_dglgraph()
    edges_srcs = graph["edge_index"][0]
    edges_dsts = graph["edge_index"][1]
    edges_etypes = graph["edge_reltype"].flatten()
    edge_referential_eids = np.arange(len(edges_srcs), dtype=np.int64)
    return edges_srcs, edges_dsts, edges_etypes, edge_referential_eids


def fetch_ogbnmag_dglgraph():
    print("loading ogbn-mag from ogb.linkproppred")
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-mag")
    graph = dataset[0]
    return graph


def fetch_ogbnmag_raw_data():
    # NB: we need to reindex nodes as nodes of any type in this data set originally starts with index 0
    # the ordering of the abosolute node indices from 0 to N-1 is author, paper, institution, field_of_study
    graph = fetch_ogbnmag_dglgraph()
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
    return create_mydgl_graph_coo_numpy(
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
