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
from .legacy_data_loader import *
import dgl

# import itertools
import torch as th

# from torch import nn
# import torch.nn.functional as F

from .. import utils
from ..RGNNUtils import *


def RGAT_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGAT")
    add_generic_RGNN_args(parser)
    args = parser.parse_args()
    return args


def RGAT_get_model(g, num_classes, hypermeters):
    embed_layer = RelGraphEmbed(
        g, hypermeters["n_infeat"], exclude=[]
    )  # exclude=["paper"])

    model = RelationalGATEncoder(
        g,
        h_dim=hypermeters["n_infeat"],
        out_dim=num_classes,
        n_heads=hypermeters["n_head"],
        num_hidden_layers=hypermeters["num_layers"] - 1,
        dropout=hypermeters["dropout"],
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
) -> tuple[HET_RelGraphEmbed, HET_RelationalGATEncoder]:
    embed_layer = HET_RelGraphEmbed(g, args.n_infeat, exclude=[])  # exclude=["paper"])

    model = HET_RelationalGATEncoder(
        g.get_num_rels(),
        h_dim=args.n_infeat,
        out_dim=num_classes,
        n_heads=args.n_head,
        num_hidden_layers=args.num_layers - 1,
        dropout=args.dropout,
        use_self_loop=True,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def RGAT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):

    # Static parameters
    # NB: default values are all moved to args
    # hyperparameters = dict(
    # num_layers=2,
    # fanout=[25, 20],
    # batch_size=1024,
    # )
    # hyperparameters.update(vars(args))
    hyperparameters = vars(args)
    print(hyperparameters)
    if not args.full_graph_training:
        assert len(args.fanout) == args.num_layers
    device = f"cuda:0" if th.cuda.is_available() else "cpu"
    # loading data
    if dgl_model_flag:
        if args.dataset != "ogbn-mag":
            raise NotImplementedError(
                "Only ogbn-mag dataset is supported for dgl model."
            )
        (g, labels, num_classes, split_idx, train_loader) = prepare_data(args)
    else:
        # (g, labels, num_classes, split_idx, train_loader) = prepare_data(hyperparameters)
        g = utils.RGNN_get_mydgl_graph(
            args.dataset,
            args.sort_by_src,
            args.sort_by_etype,
            args.reindex_eid,
            args.sparse_format,
        )
        if args.use_real_labels_and_features:
            raise NotImplementedError(
                "Not implemented loading real labels and features in utils.RGNN_get_mydgl_graph"
            )

    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    # creating model
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        if dgl_model_flag:
            labels = th.randint(0, args.num_classes, labels.shape)
        else:
            print(
                "WARNING: assuming node classification in RGAT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.get_num_nodes()])
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = RGAT_get_model(g, num_classes, hyperparameters)
    else:
        print("Using our RGAT model")
        # print(
        # int(g["original"]["col_idx"].max()) + 1,
        # )
        # print(g["original"]["row_ptr"].numel() - 1)
        embed_layer, model = RGAT_get_our_model(g, num_classes, args)
        # TODO: only certain design choices call for this. Add an option to choose.

        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        g.get_separate_node_idx_for_each_etype()
        if not args.full_graph_training:
            # need to prepare dgl graph for sampler
            g_dglgraph = g.get_dgl_graph()
            # train sampler
            # TODO: figure out split_idx train for this case
            sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
            train_loader = dgl.dataloading.NodeDataLoader(
                g_dglgraph,
                split_idx["train"],
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
        g = g.to(device)

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
                    labels,
                    # device,
                    optimizer,
                    hyperparameters,
                )
            else:
                RGNN_train_with_sampler(
                    model,
                    embed_layer(),
                    optimizer,
                    train_loader,
                    labels,
                    device,
                    hyperparameters,
                )
        else:
            if not args.full_graph_training:
                raise NotImplementedError(
                    "Not implemented full_graph_training in RGAT_main_procedure(dgl_model_flag == False)"
                )
            HET_RGNN_train_full_graph(
                g,
                model,
                embed_layer,
                optimizer,
                labels,
                device,
                hyperparameters,
            )
        # logger.print_statistics(run)

    # print("Final performance: ")
    # logger.print_statistics()


if __name__ == "__main__":
    args: argparse.Namespace = RGAT_parse_args()
    print(args)
    RGAT_main_procedure(args, dgl_model_flag=True)
