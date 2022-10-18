#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py
import numpy as np
from pathlib import Path

import torch

from ogb.nodeproppred import DglNodePropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset

import dgl
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset

DEFAULT_DIM = 64
DGL_PATH = str(Path.home()) + "/.dgl/"
torch.classes.load_library(DGL_PATH + "libgraphiler.so")

homo_dataset = {"cora": 1433, "pubmed": 500, "ppi": 50, "arxiv": 128, "reddit": 602}

hetero_dataset = ["aifb", "mutag", "bgs", "biokg", "am"]


def load_data(name, feat_dim=DEFAULT_DIM, prepare=True, to_homo=True):
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

    if name in hetero_dataset:
        if to_homo:
            g = dgl.to_homogeneous(g)

    return g, node_feats


def setup(device="cuda:0"):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


if __name__ == "__main__":
    # a place for testing data loading
    for dataset in homo_dataset:
        load_data(dataset)
    for dataset in hetero_dataset:
        load_data(dataset)
