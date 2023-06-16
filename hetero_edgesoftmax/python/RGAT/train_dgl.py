#!/usr/bin/env python3
import argparse
from .models import (
    HET_RelationalGATEncoder,
    # HET_RelationalAttLayer,
)
from .models_dgl import (
    RelationalGATEncoder,
    # RelationalAttLayer,
    # RelGraphEmbed,
    # HET_RelGraphEmbed,
)
from . import legacy_data_loader
import dgl

# import itertools
import torch as th

# from torch import nn
# import torch.nn.functional as F

from .. import utils
from ..RGNNUtils import *

from typing import Tuple


def RGAT_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGAT")
    add_generic_RGNN_args(parser, "RGAT.json", {})
    parser.add_argument("--multiply_among_weights_first_flag", action="store_true")
    parser.add_argument("--gat_edge_parallel_flag", action="store_true", default=True)
    args = parser.parse_args()
    return args


def RGAT_get_model(g, num_classes, args):
    embed_layer = RelGraphEmbed(g, args.n_infeat, exclude=[])  # exclude=["paper"])

    model = RelationalGATEncoder(
        g,
        h_dim=args.n_infeat,
        out_dim=num_classes,
        n_heads=args.n_head,
        num_hidden_layers=args.num_layers - 1,
        dropout=args.dropout,
        use_self_loop=True,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def RGAT_get_our_model(
    g: utils.MyDGLGraph, num_classes, args: argparse.Namespace
) -> Tuple[HET_RelGraphEmbed, HET_RelationalGATEncoder]:
    embed_layer = HET_RelGraphEmbed(g, args.n_infeat, exclude=[])  # exclude=["paper"])

    model = HET_RelationalGATEncoder(
        g,
        g.get_num_rels(),
        h_dim=args.n_infeat,
        out_dim=num_classes,
        n_heads=args.n_head,
        num_hidden_layers=args.num_layers - 1,
        dropout=args.dropout,
        use_self_loop=True,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
        compact_direct_indexing_flag=args.compact_direct_indexing_flag,
        multiply_among_weights_first_flag=args.multiply_among_weights_first_flag,
        gat_edge_parallel_flag=args.gat_edge_parallel_flag,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def RGAT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):
    # argument sanity check
    # Static parameters
    # NB: default values are all moved to args
    # hyperparameters = dict(
    # num_layers=2,
    # fanout=[25, 20],
    # batch_size=1024,
    # )
    # hyperparameters.update(vars(args))
    # hyperparameters = vars(args)
    # print(hyperparameters)

    # loading data
    if dgl_model_flag:
        if args.dataset != "mag" or not args.use_real_labels_and_features:
            g, _, _2 = utils.graphiler_load_data(args.dataset, to_homo=False)
        else:
            (
                g,
                labels,
                num_classes,
                split_idx,
                train_loader,
            ) = legacy_data_loader._legacy_RGAT_prepare_mag_data(args)
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
            "Not implemented loading real labels and features in utils.RGNN_get_mydgl_graph"
        )
    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        if dgl_model_flag:
            # labels = th.randint(0, args.num_classes, labels.shape)
            print(
                "WARNING: assuming node classification in RGAT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.num_nodes(t) for t in g.ntypes])
        else:
            print(
                "WARNING: assuming node classification in RGAT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.get_num_nodes()])

    # creating model and data loader
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = RGAT_get_model(g, num_classes, args)
    else:
        print("Using our RGAT model")
        embed_layer, model = RGAT_get_our_model(g, num_classes, args)
        # TODO: only certain design choices call for this. Add an option to choose.

        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        # g.generate_separate_unique_node_indices_for_each_etype()
        g.generate_separate_unique_node_indices_single_sided_for_each_etype()
        g.canonicalize_eids()
        if not args.full_graph_training:
            # need to prepare dgl graph for sampler
            g_dglgraph = g.get_dgl_graph()
            # train sampler
            # TODO: figure out split_idx train for this case
            assert len(args.fanout) == args.num_layers
            sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
            train_loader = dgl.dataloading.NodeDataLoader(
                g_dglgraph,
                split_idx["train"],
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )

    # training
    # def RGAT_main_train_procedure(
    #     g, model, embed_layer, labels, args, dgl_model_flag: bool
    # ):
    device = f"cuda:0" if th.cuda.is_available() else "cpu"
    if not dgl_model_flag:
        g = g.to(device)
        g = g.contiguous()
    embed_layer = embed_layer.to(device)
    model = model.to(device)
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
        if dgl_model_flag:
            if args.full_graph_training:
                RGNN_train_full_graph(
                    model,
                    embed_layer,
                    optimizer,
                    labels,
                    # device,
                    args,
                )
            else:
                RGNN_train_with_sampler(
                    model,
                    embed_layer(),
                    optimizer,
                    train_loader,
                    labels,
                    device,
                    args,
                )
        else:
            if not args.full_graph_training:
                raise NotImplementedError(
                    "Not implemented full_graph_training in RGAT_main_procedure(dgl_model_flag == False)"
                )
            HET_RGNN_train_full_graph(
                g.to_script_object(),
                model,
                embed_layer,
                optimizer,
                labels,
                # device,
                args,
            )
        # logger.print_statistics(run)
        # print("Final performance: ")
        # logger.print_statistics()

    # RGAT_main_train_procedure(g, model, embed_layer, labels, args, dgl_model_flag)


if __name__ == "__main__":
    args: argparse.Namespace = RGAT_parse_args()
    print(args)

    print(
        "WARNING: ignoring the hard-coded paper features in the original dataset in the original RGAT training script. This script is solely for performance R&D purposes."
    )
    RGAT_main_procedure(args, dgl_model_flag=True)
