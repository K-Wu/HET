#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py
import numpy as np
import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

GRAPHILER_DEFAULT_DIM = 64

# values are the dimension of the predefined features of the datasets provided by DGL. Notice that "proteins" are from ogbn and may not come with predefined features.
GRAPHILER_HOMO_DATASET = {
    "cora": 1433,
    "pubmed": 500,
    "ppi": 50,
    "arxiv": 128,
    "reddit": 602,
    "proteins": -1,
}

GRAPHILER_HETERO_DATASET = [
    "aifb",
    "mutag",
    "bgs",
    "am",
    "mag",
    "wikikg2",
    "fb15k",
    "biokg",
]

GRAPHILER_DATASET = GRAPHILER_HOMO_DATASET.keys() | set(
    GRAPHILER_HETERO_DATASET
)


def graphiler_load_data(
    name: str, to_homo: bool = True
) -> tuple[dgl.DGLHeteroGraph, list[int], list[tuple[int, int, int]]]:
    # feat_dim=GRAPHILER_DEFAULT_DIM,
    if name == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        g = dataset[0][0]
    elif name == "proteins":
        dataset = DglNodePropPredDataset(name="ogbn-proteins")
        g = dataset[0][0]
    elif name == "reddit":
        dataset = dgl.data.RedditDataset()
        g = dataset[0]
    elif name == "ppi":
        g = dgl.batch(
            [
                g
                for x in ["train", "test", "valid"]
                for g in dgl.data.PPIDataset(x)
            ]
        )
    elif name == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
    elif name == "pubmed":
        dataset = dgl.data.PubmedGraphDataset()
        g = dataset[0]
    elif name == "debug":
        g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))
    elif name == "aifb":
        dataset = AIFBDataset()
        g = dataset[0]
    elif name == "mutag":
        dataset = MUTAGDataset()
        g = dataset[0]
    elif name == "bgs":
        dataset = BGSDataset()
        g = dataset[0]
    elif name == "am":
        dataset = AMDataset()
        g = dataset[0]
    elif name == "mag":
        dataset = DglNodePropPredDataset(name="ogbn-mag")
        g = dataset[0][0]
    elif name == "wikikg2":
        dataset = DglLinkPropPredDataset(name="ogbl-wikikg2")
        g = dataset[0]
        src, dst = g.edges()
        reltype = torch.flatten(g.edata["reltype"]).cuda()
        num_etypes = torch.max(reltype).item() + 1
        hetero_dict = {}
        for i in range(num_etypes):
            type_index = (reltype == i).nonzero()
            hetero_dict[("n", str(i), "n")] = (
                torch.flatten(src[type_index]),
                torch.flatten(dst[type_index]),
            )
        g = dgl.heterograph(hetero_dict)
    elif name == "fb15k":
        # TODO: this is a homogeneous graph without type info
        from dgl.data import FB15k237Dataset

        dataset = FB15k237Dataset()
        g = dataset[0]
        src, dst = g.edges()
        reltype = torch.flatten(g.edata["etype"]).cuda()
        num_etypes = torch.max(reltype).item() + 1
        hetero_dict = {}
        for i in range(num_etypes):
            type_index = (reltype == i).nonzero()
            hetero_dict[("n", str(i), "n")] = (
                torch.flatten(src[type_index]),
                torch.flatten(dst[type_index]),
            )
        g = dgl.heterograph(hetero_dict)
    elif name == "biokg":
        dataset = DglLinkPropPredDataset(name="ogbl-biokg")
        g = dataset[0]
    elif name == "debug_hetero":
        g = dgl.heterograph(
            {
                ("user", "+1", "movie"): ([0, 0, 1], [0, 1, 0]),
                ("user", "-1", "movie"): ([1, 2, 2], [1, 0, 1]),
                ("user", "+1", "user"): ([0], [1]),
                ("user", "-1", "user"): ([2], [1]),
                ("movie", "+1", "movie"): ([0], [1]),
                ("movie", "-1", "movie"): ([1], [0]),
            }
        )
    else:
        raise Exception("Unknown Dataset")

    ntype_offsets = None
    canonical_etypes_id_tuple = []

    if name in GRAPHILER_HETERO_DATASET:
        ntype_dict = dict(zip(g.ntypes, range(len(g.ntypes))))
        etype_dict = dict(zip(g.etypes, range(len(g.etypes))))
        for src_type, etype, dst_type in g.canonical_etypes:
            canonical_etypes_id_tuple.append(
                (ntype_dict[src_type], etype_dict[etype], ntype_dict[dst_type])
            )

        g_homo, ntype_count, _etype_count = dgl.to_homogeneous(
            g, return_count=True
        )
        ntype_offsets = np.cumsum([0] + ntype_count).tolist()
        if to_homo:
            g = g_homo
    else:
        # homogeneous graph dataset
        canonical_etypes_id_tuple.append((0, 0, 0))
    if ntype_offsets is None:
        ntype_offsets = [0, g.number_of_nodes()]

    return (
        g,
        ntype_offsets,
        canonical_etypes_id_tuple,
    )  # returning etype for [HeteroGraphConv](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.HeteroGraphConv.html) use.
