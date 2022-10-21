from . import (
    HET_EGLRGCNSingleLayerModel,
    RGCN_main_procedure,
    RGCN_get_mydgl_graph,
    create_RGCN_parser,
)
from .. import utils
import argparse
import torch as th
import torch.nn.functional as F


def get_single_layer_model(args, mydglgraph):
    num_nodes = mydglgraph["original"]["row_ptr"].numel() - 1
    num_rels = int(mydglgraph["original"]["rel_types"].max().item()) + 1
    num_edges = mydglgraph["original"]["rel_types"].numel()
    num_classes = args.num_classes
    model = HET_EGLRGCNSingleLayerModel(
        args.input_dim,
        num_classes,
        num_rels,
        num_edges,
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
    )
    return model


def main(args):
    g = RGCN_get_mydgl_graph(args)
    model = get_single_layer_model(args, g)
    num_nodes = g["original"]["row_ptr"].numel() - 1
    feats = th.randn(num_nodes, args.input_dim)
    RGCN_main_procedure(args, g, model, feats)


if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=False)
    print(args)
    args.bfs_level = 1 + 1  # n_layers + 1 pruning used nodes for memory
    main(args)
