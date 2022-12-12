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
    add_generic_RGNN_args(parser, {})
    # parser.add_argument(
    #    "--n_hidden", type=int, default=64, help="number of hidden units"
    # )
    parser.add_argument(
        "--hgt_fused_attn_score_flag", action="store_true", default=False
    )
    parser.add_argument(
        "--fused_message_mean_aggregation_flag", action="store_true", default=True
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
        # hypermeters["n_hidden"],
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
        # args.n_hidden,
        num_classes,
        n_heads=args.n_head,
        dropout=args.dropout,
        hgt_fused_attn_score_flag=args.hgt_fused_attn_score_flag,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
        fused_message_mean_aggregation_flag=args.fused_message_mean_aggregation_flag,
    )
    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    return embed_layer, model


def HGT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):

    if dgl_model_flag:
        g, _ = utils.graphiler_load_data(args.dataset, to_homo=False)
    else:
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
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        if dgl_model_flag:
            labels = th.randint(0, args.num_classes, labels.shape)
        else:
            print(
                "WARNING: assuming node classification in RGAT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.get_num_nodes()])

    # creating model and data loader
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = HGT_get_model(g, num_classes, args)
    else:
        print("Using our RGAT model")
        # print(
        # int(g["original"]["col_idx"].max()) + 1,
        # )
        # print(g["original"]["row_ptr"].numel() - 1)
        embed_layer, model = HGT_get_our_model(g, num_classes, args)
        # TODO: only certain design choices call for this. Add an option to choose.

        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        g.get_separate_node_idx_for_each_etype()
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
    args: argparse.Namespace = HGT_parse_args()
    print(args)
    HGT_main_procedure(args, dgl_model_flag=True)
