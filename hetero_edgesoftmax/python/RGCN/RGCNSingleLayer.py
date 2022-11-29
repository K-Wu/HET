#!/usr/bin/env python3
from . import (
    HET_EGLRGCNSingleLayerModel,
    RGCN_main_procedure,
    create_RGCN_parser,
)
from .. import utils
import torch as th
import torch.nn.functional as F


def get_single_layer_model(args, mydglgraph):
    num_rels = int(mydglgraph["original"]["rel_types"].max().item()) + 1
    num_edges = mydglgraph["original"]["rel_types"].numel()
    num_classes = args.num_classes
    model = HET_EGLRGCNSingleLayerModel(
        args.input_dim,
        num_classes,
        num_rels,
        num_edges,
        args.sparse_format,
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
    )
    return model


def main(args):
    g = utils.RGNN_get_mydgl_graph(
        args.dataset,
        args.sort_by_src,
        args.sort_by_etype,
        args.reindex_eid,
        args.sparse_format,
    )
    model = get_single_layer_model(args, g)
    if args.sparse_format == "coo":
        num_nodes = int(th.max(g["original"]["row_idx"]))
    else:
        assert args.sparse_format == "csr"
        num_nodes = g["original"]["row_ptr"].numel() - 1
    num_nodes = g.get_num_nodes()
    feats = th.randn(num_nodes, args.input_dim, requires_grad=True)
    RGCN_main_procedure(args, g, model, feats)


if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=True)
    args = parser.parse_args()
    print(args)
    args.bfs_level = 1 + 1  # n_layers + 1 pruning used nodes for memory
    main(args)
