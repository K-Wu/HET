#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py
import numpy as np
from pathlib import Path
from . import loaders_from_npy
import torch

from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

GRAPHILER_DEFAULT_DIM = 64
# DGL_PATH = str(Path.home()) + "/.dgl/"
# torch.classes.load_library(DGL_PATH + "libgraphiler.so")

# values are the dimension of the predefined features of the datasets provided by DGL. Notice that "proteins" are from ogbn and may not come with predefined features.
GRAPHILER_HOMO_DATASET = {
    "cora": 1433,
    "pubmed": 500,
    "ppi": 50,
    "arxiv": 128,
    "reddit": 602,
    "proteins": -1,
}

GRAPHILER_HETERO_DATASET = ["aifb", "mutag", "bgs", "biokg", "am", "mag", "wikikg2"]


def graphiler_load_data(name, feat_dim=GRAPHILER_DEFAULT_DIM, to_homo: bool = True):
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
            [g for x in ["train", "test", "valid"] for g in dgl.data.PPIDataset(x)]
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

    node_feats = torch.rand([g.number_of_nodes(), feat_dim])

    if name in GRAPHILER_HETERO_DATASET:
        if to_homo:
            g = dgl.to_homogeneous(g)
    print(len(g.etypes))
    return (
        g,
        node_feats,
        g.etypes,
    )  # returning etype for [HeteroGraphConv](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.HeteroGraphConv.html) use.


def graphiler_load_data_as_mydgl_graph(name, feat_dim, to_homo):
    g, node_feats, g_etypes = graphiler_load_data(name, feat_dim, to_homo)
    my_g = loaders_from_npy.create_mydgl_graph_coo_from_dgl_graph(g)
    return my_g, node_feats, g.etypes


def graphiler_setup_device(device="cuda:0"):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


if __name__ == "__main__":
    # a place for testing data loading
    for dataset in GRAPHILER_HOMO_DATASET:
        graphiler_load_data(dataset)
    for dataset in GRAPHILER_HETERO_DATASET:
        graphiler_load_data(dataset)
