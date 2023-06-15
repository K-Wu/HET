#!/usr/bin/env python3

from .loaders_from_npy import *
import torch as th
from ..utils_lite import sparse_matrix_converters
from . import mydgl_graph
from . import graphiler_datasets_loader
import numpy as np


def RGNN_get_mydgl_graph(
    dataset, sort_by_src_flag, sort_by_etype_flag, no_reindex_eid_flag, sparse_format
):
    # TODO: add args for dataset, and refactor these following lines into dedicated load data function
    # load graph data
    # data_rowptr, data_colidx, data_reltypes, data_eids
    # transposed_data_rowptr, transposed_data_colidx, transposed_data_reltypes, transposed_data_eids,

    dataset_sort_flag = sort_by_src_flag or sort_by_etype_flag

    if no_reindex_eid_flag:
        raise NotImplementedError(
            "reindex eid is currently 1) a must for logic like inverse idx and eid indirection omittance, and 2) enabled anyway in graphiler loader"
        )

    if dataset == "fb15k":
        print("WARNING - loading fb15k. Currently we only support a few dataset.")
        (
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
        ) = load_fb15k237(
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
        ntype_offsets = [0, 14541]
        canonical_etype_indices_tuples = []
        for idx_etype in range(int(max(edge_etypes)) + 1):
            canonical_etype_indices_tuples.append((0, idx_etype, 0))
    elif dataset == "wikikg2":
        print("WARNING - loading wikikg2. Currently we only support a few dataset.")
        (
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
        ) = load_wikikg2(
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
        ntype_offsets = [0, 2500604]  # there is only one node type
        canonical_etype_indices_tuples = []
        for idx_etype in range(int(max(edge_etypes)) + 1):
            canonical_etype_indices_tuples.append((0, idx_etype, 0))
    elif dataset in graphiler_datasets_loader.GRAPHILER_DATASET:
        print("RGNN_get_mydgl_graph loading graphiler dataset")
        (
            g,
            ntype_offsets,
            canonical_etype_indices_tuples,
        ) = graphiler_datasets_loader.graphiler_load_data_as_mydgl_graph(dataset, True)
        edge_srcs, edge_dsts, edge_etypes, edge_referential_eids = (
            g["original"]["row_indices"],
            g["original"]["col_indices"],
            g["original"]["rel_types"],
            g["original"]["eids"],
        )

        if len(ntype_offsets) > 2 and sort_by_src_flag:
            print(
                "WARNING! >1 ntypes while sort_by_src_flag is set True. Experiemental sort within each node type interval is called"
            )

        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = (edge_srcs, edge_dsts, edge_etypes, edge_referential_eids)

        (
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
        ) = sort_coo_according_to_flags(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            ntype_offsets,
            dataset_sort_flag,
            sort_by_src_flag,
            torch_flag=True,
            infidel_sort_flag=False,
        )

        (
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        ) = sort_coo_according_to_flags(
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
            ntype_offsets,
            dataset_sort_flag,
            sort_by_src_flag,
            torch_flag=True,
            infidel_sort_flag=False,
        )
    else:
        print(
            "ERROR! now only support fb15k, wikikg2 and those in graphiler datasets. Loading it now"
        )
        exit(1)
    if no_reindex_eid_flag:
        raise NotImplementedError(
            "reindex eid is currently a must for logic like inverse idx and eid indirection omittance"
        )
    else:
        edge_new_eids = (
            np.arange(edge_referential_eids.shape[0]).astype(np.int64).tolist()
        )
        edge_referential_to_new_eids_mapping = dict(
            zip(edge_referential_eids.tolist(), edge_new_eids)
        )
        transposed_edge_new_eids = np.array(
            list(
                map(
                    edge_referential_to_new_eids_mapping.__getitem__,
                    transposed_edge_referential_eids.tolist(),
                )
            )
        ).astype(np.int64)
        if th.is_tensor(edge_srcs):
            edge_referential_eids = th.tensor(edge_new_eids)
            transposed_edge_referential_eids = th.tensor(transposed_edge_new_eids)

    if sparse_format == "coo":
        g = create_mydgl_graph_coo_with_transpose(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            transposed_edge_srcs,
            transposed_edge_dsts,
            transposed_edge_etypes,
            transposed_edge_referential_eids,
        )
        g.sequential_eids_format = "original_coo"
    elif sparse_format == "csr":
        # coo to csr conversion
        (
            data_rowptr,
            data_colidx,
            data_reltypes,
            data_eids,
        ) = sparse_matrix_converters.coo2csr(
            edge_srcs,
            edge_dsts,
            edge_etypes,
            edge_referential_eids,
            torch_flag=th.is_tensor(edge_srcs),
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
            torch_flag=th.is_tensor(edge_srcs),
        )
        # create graph
        g = create_mydgl_graph_csr_with_transpose(
            data_rowptr,
            data_colidx,
            data_reltypes,
            data_eids,
            transposed_data_rowptr,
            transposed_data_colidx,
            transposed_data_reltypes,
            transposed_data_eids,
        )

        g.sequential_eids_format = "original_coo"
    else:
        raise NotImplementedError("sparse format not supported")
    g["original"]["node_type_offsets"] = th.LongTensor(ntype_offsets)
    return g, canonical_etype_indices_tuples


def convert_mydgl_graph_csr_to_coo(g):
    # we haven't implemented csr2coo for tensors so we need to convert to numpy first
    row_ptr = g["original"]["row_ptrs"].numpy()
    col_idx = g["original"]["col_indices"].numpy()
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
            g["transposed"]["row_ptrs"].numpy(),
            g["transposed"]["col_indices"].numpy(),
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
        g["original"]["row_ptrs"],
        g["original"]["col_indices"],
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
            g["transposed"]["row_ptrs"],
            g["transposed"]["col_indices"],
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
    g["original"]["row_ptrs"] = row_ptr
    g["original"]["col_indices"] = col_idx
    g["original"]["rel_types"] = rel_types
    g["original"]["eids"] = eids
    g["transposed"] = dict()
    g["transposed"]["row_ptrs"] = transposed_row_ptr
    g["transposed"]["col_indices"] = transposed_col_idx
    g["transposed"]["rel_types"] = transposed_rel_types
    g["transposed"]["eids"] = transposed_eids
    return g


def create_mydgl_graph_csr_torch(row_ptr, col_idx, rel_types, eids):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_ptrs"] = row_ptr
    g["original"]["col_indices"] = col_idx
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


def create_mydgl_graph_csr_with_transpose(
    row_ptr,
    col_idx,
    rel_types,
    eids,
    transposed_row_ptr,
    transposed_col_idx,
    transposed_rel_types,
    transposed_eids,
):
    if th.is_tensor(row_ptr):
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
    else:
        return create_mydgl_graph_csr_with_transpose_numpy(
            row_ptr,
            col_idx,
            rel_types,
            eids,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_rel_types,
            transposed_eids,
        )


def create_mydgl_graph_coo_with_transpose(
    row_ptr,
    col_idx,
    rel_types,
    eids,
    transposed_row_ptr,
    transposed_col_idx,
    transposed_rel_types,
    transposed_eids,
):
    if th.is_tensor(row_ptr):
        return create_mydgl_graph_coo_with_transpose_torch(
            row_ptr,
            col_idx,
            rel_types,
            eids,
            transposed_row_ptr,
            transposed_col_idx,
            transposed_rel_types,
            transposed_eids,
        )
    else:
        return create_mydgl_graph_coo_with_transpose_numpy(
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
def create_mydgl_graph_coo_from_hetero_dgl_graph(g):
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
        # add to total
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
    mydgl_graph.sequential_eids_format = "original_coo"
    mydgl_graph.import_metadata_from_dgl_heterograph(g)
    return mydgl_graph


@th.no_grad()
def create_mydgl_graph_coo_from_homo_dgl_graph(g, dataset_originally_homo_flag):
    total_edge_srcs, total_edge_dsts = g.edges()
    if dataset_originally_homo_flag:
        etypes = th.zeros(g.number_of_edges(), dtype=th.int64)
    else:
        assert "_TYPE" in g.edata, "Heterogeneous graph must have _TYPE edge data"
        etypes = g.edata["_TYPE"]
    mydgl_graph = create_mydgl_graph_coo_torch(
        total_edge_srcs,
        total_edge_dsts,
        etypes,
        th.arange(g.number_of_edges(), dtype=th.int64),
    )
    mydgl_graph.sequential_eids_format = "original_coo"
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
    g["original"]["row_indices"] = edge_srcs
    g["original"]["col_indices"] = edge_dsts
    g["original"]["rel_types"] = edge_etypes
    g["original"]["eids"] = edge_eids
    g["transposed"] = dict()
    g["transposed"]["row_indices"] = transposed_edge_srcs
    g["transposed"]["col_indices"] = transposed_edge_dsts
    g["transposed"]["rel_types"] = transposed_edge_etypes
    g["transposed"]["eids"] = transposed_edge_eids
    return g


def create_mydgl_graph_coo_torch(
    edge_srcs, edge_dsts, edge_etypes, edge_referential_eids
):
    g = mydgl_graph.MyDGLGraph()
    g["original"] = dict()
    g["original"]["row_indices"] = edge_srcs
    g["original"]["col_indices"] = edge_dsts
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
