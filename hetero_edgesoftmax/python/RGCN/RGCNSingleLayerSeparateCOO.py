#!/usr/bin/env python3
from . import (
    HET_EGLRGCNSingleLayerModel,
    RGCN_main_procedure,
    create_RGCN_parser,
)
from .. import utils
import torch as th
import torch.nn.functional as F


def get_single_layer_separate_coo_model(args, mydglgraph):
    num_rels = int(mydglgraph["original"]["rel_types"].max().item()) + 1
    num_edges = mydglgraph["original"]["rel_types"].numel()
    num_classes = args.num_classes
    print("num_nodes", mydglgraph.get_num_nodes())
    print("num_edges ", num_edges)
    print("num_classes ", num_classes)
    print("num_rels ", num_rels)
    model = HET_EGLRGCNSingleLayerModel(
        args.n_infeat,
        num_classes,
        num_rels,
        num_edges,
        "separate_coo",
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
        hybrid_assign_flag=args.hybrid_assign_flag,
        num_blocks_on_node_forward=args.num_blocks_on_node_forward,
        num_blocks_on_node_backward=args.num_blocks_on_node_backward,
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
    model = get_single_layer_separate_coo_model(args, g)
    if args.sparse_format == "coo":
        num_nodes = int(th.max(g["original"]["row_idx"]))
    else:
        assert args.sparse_format == "csr"
        num_nodes = g["original"]["row_ptr"].numel() - 1
    num_nodes = g.get_num_nodes()
    feats = th.randn(num_nodes, args.n_infeat, requires_grad=True)

    g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
    g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
    RGCN_main_procedure(args, g, model, feats)


if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=True)
    args = parser.parse_args()
    print(args)
    args.bfs_level = 1 + 1  # num_layers + 1 pruning used nodes for memory
    main(args)
