#!/usr/bin/env python3
from __future__ import annotations
from .. import utils
from ..RGNNUtils import *

import argparse
import dgl
import torch as th
from .models import *
from .models_dgl import *

from ..utils import MyDGLGraph


def HGT_get_and_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HGT")
    add_generic_RGNN_args(parser, "HGT.json", {})
    # parser.add_argument(
    #    "--n_hidden", type=int, default=64, help="number of hidden units"
    # )
    parser.add_argument(
        "--hgt_fused_attn_score_flag", action="store_true", default=False
    )
    parser.add_argument(
        "--fused_message_mean_aggregation_flag",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--multiply_among_weights_first_flag",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def HGT_get_model(g: dgl.DGLGraph, num_classes: int, args: argparse.Namespace):
    embed_layer = RelGraphEmbed(g, args.n_infeat, exclude=[])
    node_dict = {}
    edge_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in g.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
    model = HGT_DGLHetero(
        node_dict,
        edge_dict,
        args.n_infeat,
        # hypermeters["n_hidden"],
        num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    print(embed_layer)
    print(
        "Number of embedding parameters:"
        f" {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(
        "Number of model parameters:"
        f" {sum(p.numel() for p in model.parameters())}"
    )
    # NB: when transfering model to GPU, notice that DGLGraph g also needs to be transfered to GPU
    return embed_layer, model


def HGT_get_our_model(
    g: MyDGLGraph,
    canonical_etype_indices_tuples: list[tuple[int, int, int]],
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[HET_RelGraphEmbed, HET_HGT_DGLHetero]:
    embed_layer = HET_RelGraphEmbed(g.get_num_nodes(), args.n_infeat)
    model = HET_HGT_DGLHetero(
        g.get_num_ntypes(),
        g.get_num_rels(),
        args.n_infeat,
        # args.n_hidden,
        num_classes,
        torch.tensor(
            list(zip(*canonical_etype_indices_tuples))[0],
            dtype=torch.long,
            requires_grad=False,
        ),
        torch.tensor(
            list(zip(*canonical_etype_indices_tuples))[2],
            dtype=torch.long,
            requires_grad=False,
        ),
        num_heads=args.num_heads,
        dropout=args.dropout,
        multiply_among_weights_first_flag=args.multiply_among_weights_first_flag,
        hgt_fused_attn_score_flag=args.hgt_fused_attn_score_flag,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
        compact_direct_indexing_flag=args.compact_direct_indexing_flag,
        fused_message_mean_aggregation_flag=args.fused_message_mean_aggregation_flag,
    )
    print(embed_layer)
    print(
        "Number of embedding parameters:"
        f" {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(
        "Number of model parameters:"
        f" {sum(p.numel() for p in model.parameters())}"
    )
    return embed_layer, model


def HGT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):
    """dgl_model_flag determines if the model execution uses dgl or HET. For train_dgl.py, it uses dgl_model_flag=True. And for train.py, it uses dgl_model_flag=False."""
    if dgl_model_flag:
        g, _, _2 = utils.graphiler_load_data(args.dataset, to_homo=False)
    else:
        g, canonical_etype_indices_tuples = utils.RGNN_get_mydgl_graph(
            args.dataset,
            args.sort_by_src,
            args.sort_by_etype,
            args.no_reindex_eid,
            args.sparse_format,
        )

    if args.use_real_labels_and_features:
        raise NotImplementedError(
            "Not implemented loading real labels and features in"
            " utils.RGNN_get_mydgl_graph"
        )
    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        if dgl_model_flag:
            labels = th.randint(0, args.num_classes, labels.shape)
        else:
            print(
                "WARNING: assuming node classification in"
                " HGT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.get_num_nodes()])

    # Creating model and data loader
    if dgl_model_flag:
        print("Using DGL HGT model")
        embed_layer, model = HGT_get_model(g, num_classes, args)
    else:
        print("Using our HGT model")

        # TODO: only certain design choices call for this. Add an option to choose.

        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        g.generate_separate_unique_node_indices_for_each_etype()
        g.generate_separate_unique_node_indices_single_sided_for_each_etype()
        g.canonicalize_eids()

        # Training
        device = f"cuda:0" if th.cuda.is_available() else "cpu"
        # This operation is effective because reference to g is stored in model and this operation does to() in place, i.e., without creating new g, all tensors as values of g's dictionaries is replaced with new tensors on device, while the keys stay the same.
        g.to_(device)
        g.contiguous_()

        # execute g = g.to_script_object() here so that 1) the script object veresion is stored as model.mydglgraph, and succeeding operation on g after get_our_model is applicable to model.mydglgraph
        # TODO: fix this in future if it breaks
        # g = g.to_script_object()

        embed_layer, model = HGT_get_our_model(
            g, canonical_etype_indices_tuples, num_classes, args
        )

    if not args.full_graph_training:
        # need to prepare dgl graph for sampler
        # g_dglgraph = g.get_dgl_graph()
        g_dglgraph_hetero, _, _2 = utils.graphiler_load_data(
            args.dataset, to_homo=False
        )
        num_of_nodes = sum(
            [
                g_dglgraph_hetero.number_of_nodes(ntype)
                for ntype in g_dglgraph_hetero.ntypes
            ]
        )
        train_loader = dgl.dataloading.DataLoader(
            g_dglgraph_hetero,
            list(range(num_of_nodes)),
            dgl.dataloading.MultiLayerFullNeighborSampler(1),
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )

    # Training
    device = f"cuda:0" if th.cuda.is_available() else "cpu"
    # if not dgl_model_flag:
    #     # This operation is effective because reference to g is stored in model and this operation does to() in place, i.e., without creating new g, all tensors as values of g's dictionaries is replaced with new tensors on device, while the keys stay the same.
    #     g.to(device)
    #     g.contiguous()
    embed_layer = embed_layer.to(device)
    model = model.to(device)
    model.const_to(device)
    labels = labels.to(device)
    for run in range(args.runs):
        embed_layer.reset_parameters()
        model.reset_parameters()

        # optimizer
        all_params = [*model.parameters()] + [
            *embed_layer.parameters()
        ]  # itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr)
        print(f"Run: {run + 1:02d}, ")
        if dgl_model_flag:  # Use vanilla DGL to execute the model
            assert len(args.fanout) == args.num_layers
            if args.full_graph_training:
                RGNN_train_full_graph(
                    model,
                    embed_layer,
                    labels,
                    # device,
                    optimizer,
                    args,
                )
            else:  # Use HET to execute the model
                RGNN_train_with_sampler(
                    model,
                    embed_layer(),
                    optimizer,
                    train_loader,
                    labels,
                    device,
                    args,
                )
        else:  # Not dgl_model_flag
            # Type annotation
            assert isinstance(embed_layer, HET_RelGraphEmbed)
            if args.full_graph_training:
                HET_RGNN_train_full_graph(
                    g,
                    model,
                    embed_layer,
                    optimizer,
                    labels,
                    # device,
                    args,
                )
            else:
                HET_RGNN_train_with_sampler(
                    g,
                    train_loader,
                    model,
                    embed_layer,
                    optimizer,
                    labels,
                    args,
                )


if __name__ == "__main__":
    args: argparse.Namespace = HGT_get_and_parse_args()
    print(args)
    HGT_main_procedure(args, dgl_model_flag=True)
