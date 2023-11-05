#!/usr/bin/env python3
from . import (
    HET_EGLRGCNSingleLayerModel,
    RGCN_main_procedure,
    create_RGCN_parser,
    RGCN_prepare_data,
)
from .. import utils
from .. import utils_lite
from ..RGNNUtils import HET_RelGraphEmbed
import torch as th
import torch.nn.functional as F


def get_single_layer_model(args, mydglgraph):
    num_rels = int(mydglgraph["original"]["rel_types"].max().item()) + 1
    num_nodes = mydglgraph.get_num_nodes()
    num_edges = mydglgraph["original"]["rel_types"].numel()
    num_classes = args.num_classes
    print("num_nodes", num_nodes)
    print("num_edges ", num_edges)
    print("num_classes ", num_classes)
    print("num_rels ", num_rels)
    # TODO: pass num_heads if applicable to RGCN
    model = HET_EGLRGCNSingleLayerModel(
        mydglgraph,
        args.n_infeat,
        num_classes,
        num_rels,
        num_nodes,
        num_edges,
        args.sparse_format,
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
        hybrid_assign_flag=args.hybrid_assign_flag,
        num_blocks_on_node_forward=args.num_blocks_on_node_forward,
        num_blocks_on_node_backward=args.num_blocks_on_node_backward,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
        compact_direct_indexing_flag=args.compact_direct_indexing_flag,
    )
    return model


# TODO: apply changes to main_seastar() in RGCN.py to this function
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
        # g.separate_coo_rel_ptrs_cpu_contiguous = g.graph_data["separate"]["coo"]["original"]["rel_ptrs"].cpu().contiguous()

    g.canonicalize_eids()
    g = g.cuda().contiguous()
    # Execute g = g.to_script_object() here so that 1) the script object veresion is stored as model.mydglgraph, and succeeding operation on g after get_our_model is applicable to model.mydglgraph
    # TODO: fix this in future if it breaks
    # g = g.to_script_object()

    # TODO: need to move g to cuda and then store g in the model
    model = get_single_layer_model(args, g)
    # num_nodes = g.get_num_nodes()
    # feats = th.randn(
    #     num_nodes,
    #     args.n_infeat,
    #     requires_grad=False,
    #     device=th.device(f"cuda:{args.gpu}"),
    # )
    node_embed_layer = HET_RelGraphEmbed(g, args.n_infeat)
    node_embed_layer = node_embed_layer.to(th.device(f"cuda:{args.gpu}"))
    # feats = node_embed_layer()

    # g = g.to_script_object()

    (
        labels,
        edge_norm,
        node_embed_layer,
        model,
        optimizer,
    ) = RGCN_prepare_data(args, g, model, node_embed_layer)

    RGCN_main_procedure(
        model, node_embed_layer, optimizer, labels, args, edge_norm
    )


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
