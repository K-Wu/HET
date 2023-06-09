#!/usr/bin/env python3
from . import (
    HET_EGLRGCNSingleLayerModel,
    RGCN_main_procedure,
    create_RGCN_parser,
)
from .. import utils
from .. import utils_lite
import torch as th
import torch.nn.functional as F


def get_single_layer_model(args, mydglgraph):
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
        args.sparse_format,
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
        hybrid_assign_flag=args.hybrid_assign_flag,
        num_blocks_on_node_forward=args.num_blocks_on_node_forward,
        num_blocks_on_node_backward=args.num_blocks_on_node_backward,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
    )
    return model


def RGCNSingleLayer_main(args):
    if args.sparse_format == "separate_coo":
        mydgl_graph_format = "csr"
    else:
        mydgl_graph_format = args.sparse_format
        assert args.sparse_format == "coo" or args.sparse_format == "csr"
    g, canonical_etype_indices_tuples = utils.RGNN_get_mydgl_graph(
        args.dataset,
        args.sort_by_src,
        args.sort_by_etype,
        args.no_reindex_eid,
        mydgl_graph_format
        # args.sparse_format,
    )
    model = get_single_layer_model(args, g)
    num_nodes = g.get_num_nodes()
    feats = th.randn(num_nodes, args.n_infeat, requires_grad=True)
    if args.sparse_format == "separate_coo":
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        if args.compact_as_of_node_flag:
            g.generate_separate_unique_node_indices_for_each_etype()
            g.generate_separate_unique_node_indices_single_sided_for_each_etype()
            print(
                "size of unique nodes",
                g.get_separate_unique_node_indices()["node_indices"].shape,
                g.get_separate_unique_node_indices_single_sided()[
                    "node_indices_row"
                ].shape,
                g.get_separate_unique_node_indices_single_sided()[
                    "node_indices_col"
                ].shape,
            )
    g.canonicalize_eids()
    RGCN_main_procedure(args, g, model, feats)


if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=True)
    args = parser.parse_args()
    print(args)
    args.bfs_level = 1 + 1  # num_layers + 1 pruning used nodes for memory
    if args.dataset == "all":
        for dataset in utils_lite.GRAPHILER_HETERO_DATASET:
            args.dataset = dataset
            print(f"Training on {dataset}")
            RGCNSingleLayer_main(args)
    else:
        RGCNSingleLayer_main(args)
