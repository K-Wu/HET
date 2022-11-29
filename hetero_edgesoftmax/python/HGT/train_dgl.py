#!/usr/bin/env python3
from .. import utils
from ..RGNNUtils import *

import argparse
import dgl
import torch as th
from .models import *
from .models_dgl import *


def HGT_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HGT")
    add_generic_RGNN_args(parser)
    parser.add_argument(
        "--n_hidden", type=int, default=64, help="number of hidden units"
    )
    args = parser.parse_args()
    return args


def HGT_get_model(g: dgl.DGLGraph, num_classes, hypermeters):
    embed_layer = RelGraphEmbed(g, hypermeters["n_infeat"], exclude=[])
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
    model = HGT_DGLHetero(
        node_dict,
        edge_dict,
        hypermeters["n_infeat"],
        hypermeters["n_hidden"],
        num_classes,
        n_heads=hypermeters["n_head"],
        dropout=hypermeters["dropout"],
    )
    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    # NB: when transfering model to GPU, notice that DGLGraph g also needs to be transfered to GPU
    return embed_layer, model


def HGT_get_our_model(
    g: utils.MyDGLGraph, num_classes, args: argparse.Namespace
) -> tuple[HET_RelGraphEmbed, HET_HGT_DGLHetero]:
    embed_layer = HET_RelGraphEmbed(g, args.n_infeat, exclude=[])
    model = HET_HGT_DGLHetero(
        g,
        args.n_infeat,
        args.n_hidden,
        num_classes,
        n_heads=args.n_head,
        dropout=args.dropout,
    )
    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    return embed_layer, model


def HGT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):
    pass


if __name__ == "__main__":
    args: argparse.Namespace = HGT_parse_args()
    print(args)
    HGT_main_procedure(args, dgl_model_flag=True)
