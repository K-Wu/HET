#!/usr/bin/env python3
from __future__ import annotations
from typing import Union, Iterable, Callable
import torch as th
import torch.nn.functional as F
from torch import nn
import dgl

from typing import NoReturn

from torch.cuda import (
    memory_allocated,
    max_memory_allocated,
    reset_peak_memory_stats,
)

# import torch.jit
from .. import utils_lite

import nvtx

from ..utils import (
    MyDGLGraph,
    get_mydgl_graph_dataloader,
    get_funcs_to_propagate_and_produce_metadata,
)
from dgl.heterograph import DGLBlock
from dgl import DGLHeteroGraph
from dgl.dataloading import DataLoader
import argparse
import numpy as np

import torch.jit


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph.

    Parameters
    ----------RelGraphEmbed
    g : DGLGraph
        Input graph.
    embed_size : int
        The length of each embedding vector
    exclude : list[str]
        The list of node-types to exclude (e.g., because they have natural features)
    """

    @utils_lite.warn_default_arguments
    def __init__(self, g: DGLHeteroGraph, embed_size, exclude=list()):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size

        # create learnable embeddings for all nodes, except those with a node-type in the "exclude" list
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if ntype in exclude:
                continue
            embed = nn.Parameter(
                th.Tensor(g.number_of_nodes(ntype), self.embed_size)
            )
            self.embeds[ntype] = embed

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeds.values():
            nn.init.xavier_uniform_(emb)

    def forward(self, indices=None, block: Union[None, DGLBlock] = None):
        if indices is None:
            return self.embeds
        else:
            return {k: self.embeds[k][indices[k]] for k in indices}


class HET_RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph.

    Parameters
    ----------RelGraphEmbed
    num_nodes : int
        Number of nodes of the input graph.
    embed_size : int
        The length of each embedding vector
    exclude : list[str]
        The list of node-types to exclude (e.g., because they have natural features)
    """

    @utils_lite.warn_default_arguments
    def __init__(
        self, num_nodes: int, embed_size, dtype=th.float32, requires_grad=True
    ):
        super(HET_RelGraphEmbed, self).__init__()
        self.embed_size = embed_size

        # dtype and requires_grad are introduced to support special cases such as seastar RGCN (int64 and False)
        if dtype == th.int64:
            assert embed_size == 1, (
                "embed_size must be 1 for int64 as this is specifically"
                " provided for seastar RGCN"
            )
            data = torch.arange(num_nodes)
        elif dtype == th.float32:
            data = th.Tensor(num_nodes, self.embed_size)
        else:
            raise ValueError("dtype not supported")
        self.embeds = nn.Parameter(data, requires_grad=requires_grad)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embeds)

    def forward(self, indices=None, block: Union[None, DGLBlock] = None):
        # indices: embedding lookup in minibatch training/inference
        if indices is None:
            return self.embeds
        else:
            return self.embeds[indices]


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb


def HET_RGNN_train_with_sampler(*args, **kwargs) -> NoReturn:
    raise NotImplementedError(
        "In HET, minibatch and full-graph use the same training procedure"
        " function."
    )


def get_labels(labels: th.Tensor, output_node_idxes: th.Tensor | None):
    if output_node_idxes is None:
        return labels
    else:
        return labels[output_node_idxes]


def HET_RGNN_train_full_graph(
    g: MyDGLGraph,
    model,
    node_embed_layer: HET_RelGraphEmbed,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
    # TODO: add optional argument edge_norm to support RGCN
):
    return HET_RGNN_train(
        lambda: [(None, None, g)],
        model,
        node_embed_layer,
        optimizer,
        labels,
        args,
    )


def HET_RGNN_train_with_sampler(
    g: MyDGLGraph,
    dataloader: dgl.dataloading.DataLoader,
    model,
    node_embed_layer,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
):
    """
    An example of dataloader is
    dataloader = dgl.dataloading.DataLoader(
        g_dgl_homo,
        list(range(g_dgl_homo.number_of_nodes())),
        dgl.dataloading.MultiLayerFullNeighborSampler(1),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )
    """
    funcs_to_apply = get_funcs_to_propagate_and_produce_metadata(g)
    get_new_dataloader = lambda: get_mydgl_graph_dataloader(
        dataloader, funcs_to_apply
    )
    return HET_RGNN_train(
        get_new_dataloader,
        model,
        node_embed_layer,
        optimizer,
        labels,
        args,
    )


def HET_RGNN_train(
    get_new_dataloader: Callable[
        [],
        Iterable[tuple[None | th.Tensor, None | th.Tensor, MyDGLGraph]],
    ],
    model,
    node_embed_layer,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
    # TODO: add optional argument edge_norm to support RGCN
):
    # training loop

    if args.jit_script_enabled:
        model = torch.jit.script(model)

    forward_time = []
    backward_time = []
    training_time = []

    warm_up_forward_intermediate_memory = []
    warm_up_intermediate_memory = []

    model.train()
    model.requires_grad_(True)
    node_embed_layer.train()
    node_embed_layer.requires_grad_(True)

    memory_offset = torch.cuda.memory_allocated()
    # warm up
    if not args.no_warm_up:
        for epoch in range(5):
            model.train()
            model.requires_grad_(True)
            node_embed_layer.train()
            node_embed_layer.requires_grad_(True)
            dataloader = get_new_dataloader()
            for input_nodes, output_nodes, g in dataloader:
                all_nodes = (
                    torch.cat([output_nodes, input_nodes])
                    if (output_nodes is not None)
                    else None
                )
                print(
                    "Graph stats: ",
                    g.get_num_edges(),
                    g.get_num_nodes(),
                    g.get_num_rels(),
                )

                optimizer.zero_grad()
                node_embed = node_embed_layer(all_nodes)
                logits = model(g, node_embed)
                # forward_intermediate_memory needs to be collected before backward
                warm_up_forward_intermediate_memory.append(
                    torch.cuda.max_memory_allocated() - memory_offset
                )
                y_hat = logits.log_softmax(dim=-1)
                loss = F.nll_loss(y_hat, labels)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                warm_up_intermediate_memory.append(
                    torch.cuda.max_memory_allocated() - memory_offset
                )
                # Reset the memory stat to clear up the maximal memory allocated stat.
                reset_peak_memory_stats()
    print("start training...")

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch:02d}")
        import contextlib

        model.train()
        model.requires_grad_(True)
        node_embed_layer.train()
        node_embed_layer.requires_grad_(True)

        dataloader = get_new_dataloader()
        for idx_step, (input_nodes, output_nodes, g) in enumerate(dataloader):
            all_nodes = (
                torch.cat([output_nodes, input_nodes])
                if (output_nodes is not None)
                else None
            )
            node_embed = node_embed_layer(all_nodes)
            curr_labels = get_labels(labels, all_nodes)

            optimizer.zero_grad()
            th.cuda.synchronize()
            # nvtx.push_range("inference", domain="my_domain")
            forward_prop_start = th.cuda.Event(enable_timing=True)
            forward_prop_end = th.cuda.Event(enable_timing=True)
            forward_prop_start.record()
            # for idx in range(10):
            logits = model(g, node_embed)
            # logits = scripted_model(node_embed)
            # logits = model(emb, blocks)
            forward_prop_end.record()
            th.cuda.synchronize()

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, curr_labels)

            backward_prop_start = th.cuda.Event(enable_timing=True)
            backward_prop_end = th.cuda.Event(enable_timing=True)
            backward_prop_start.record()

            loss.backward()
            optimizer.step()
            backward_prop_end.record()
            th.cuda.synchronize()
            # nvtx.pop_range()

            # TODO: should be # edges when training full graph
            # total_loss += loss.item() * args.batch_size

            # result = test(g, model, node_embed, labels, device, split_idx, args)
            # logger.add_result(run, result)
            # train_acc, valid_acc, test_acc = result
            print(
                f"Epoch: {epoch + 1 :02d}, Step: {idx_step + 1 :02d},"
                f"Loss (w/o dividing sample num): {loss:.4f}, "
            )
            print(
                "Forward prop time:"
                f" {forward_prop_start.elapsed_time(forward_prop_end)} ms"
            )
            print(
                "Backward prop time:"
                f" {backward_prop_start.elapsed_time(backward_prop_end)} ms"
            )
            print(
                "Total time:"
                f" {forward_prop_start.elapsed_time(backward_prop_end)} ms"
            )
            if epoch >= 3:
                forward_time.append(
                    forward_prop_start.elapsed_time(forward_prop_end)
                )
                backward_time.append(
                    backward_prop_start.elapsed_time(backward_prop_end)
                )
                training_time.append(
                    forward_prop_start.elapsed_time(backward_prop_end)
                )

    print(
        "Epochs: Max memory allocated (MB) ",
        torch.cuda.max_memory_allocated() / 1024 / 1024,
    )
    print(
        "Epochs: Intermediate memory allocated (MB) ",
        (torch.cuda.max_memory_allocated() - memory_offset) / 1024 / 1024,
    )
    if not args.no_warm_up:
        print(
            "WarmUp: Intermediate memory allocated (MB) ",
            max(warm_up_intermediate_memory) / 1024 / 1024,
        )
        print(
            "WarmUp: Forward intermediate memory allocated (MB) ",
            max(warm_up_forward_intermediate_memory) / 1024 / 1024,
        )
    if len(forward_time[len(forward_time) // 4 :]) == 0:
        print(
            "insufficient run to report mean time. skipping. (in the json it"
            " might show as nan)"
        )
    else:
        print(
            "Mean forward time: {:4f} ms".format(
                np.mean(forward_time[len(forward_time) // 4 :])
            )
        )
        print(
            "Mean backward time: {:4f} ms".format(
                np.mean(backward_time[len(backward_time) // 4 :])
            )
        )
        print(
            "Mean training time: {:4f} ms".format(
                np.mean(training_time[len(training_time) // 4 :])
            )
        )

    # write to file
    import json

    if args.logfile_enabled:
        with open(args.logfilename, "a") as fd:
            json.dump(
                {
                    "dataset": args.dataset,
                    "mean_forward_time": np.mean(
                        forward_time[len(forward_time) // 4 :]
                    ),
                    "mean_backward_time": np.mean(
                        backward_time[len(backward_time) // 4 :]
                    ),
                    "mean_training_time": np.mean(
                        training_time[len(training_time) // 4 :]
                    ),
                    "forward_time": forward_time,
                    "backward_time": backward_time,
                    "training_time": training_time,
                    "max_memory_usage (mb)": (
                        torch.cuda.max_memory_allocated()
                    )
                    / 1024
                    / 1024,
                    "epochs_intermediate_memory_usage (mb)": (
                        torch.cuda.memory_allocated() - memory_offset
                    )
                    / 1024
                    / 1024,
                    "warm_up_intermediate_memory_usage (mb)": (
                        [
                            ele / 1024 / 1024
                            for ele in warm_up_intermediate_memory
                        ]
                    ),
                    "warm_up_forward_intermediate_memory_usage (mb)": (
                        [
                            ele / 1024 / 1024
                            for ele in warm_up_forward_intermediate_memory
                        ]
                    ),
                },
                fd,
            )
            fd.write("\n")

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGNN_train_full_graph(
    model,
    node_embed: RelGraphEmbed,
    optimizer,
    labels,
    args: argparse.Namespace,
):
    # training loop
    print("start training...")
    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0

        emb = node_embed

        lbl = labels

        if th.cuda.is_available():
            emb = {k: e.cuda() for k, e in emb.items()}
            lbl = {k: e.cuda() for k, e in lbl.items()}
        optimizer.zero_grad()
        forward_prop_start = th.cuda.Event(enable_timing=True)
        forward_prop_end = th.cuda.Event(enable_timing=True)
        forward_prop_start.record()
        logits = model(emb)
        forward_prop_end.record()
        th.cuda.synchronize()
        # logits = model(emb, blocks)
        loss = None
        for category in logits:
            y_hat = logits[category].log_softmax(dim=-1)
            if loss is None:
                loss = F.nll_loss(y_hat, lbl)
            else:
                loss += F.nll_loss(y_hat, lbl)

        backward_prop_start = th.cuda.Event(enable_timing=True)
        backward_prop_end = th.cuda.Event(enable_timing=True)
        backward_prop_start.record()
        loss.backward()
        optimizer.step()
        backward_prop_end.record()
        th.cuda.synchronize()

        # TODO: should be # edges when training full graph
        total_loss += loss.item() * args.batch_size

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, "
            f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
        print(
            "Forward prop time:"
            f" {forward_prop_start.elapsed_time(forward_prop_end)} ms"
        )
        print(
            "Backward prop time:"
            f" {backward_prop_start.elapsed_time(backward_prop_end)} ms"
        )

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGNN_train_with_sampler(
    model,
    node_embed: RelGraphEmbed,
    optimizer,
    train_loader: DataLoader,
    labels,
    device,
    args: argparse.Namespace,
):
    # training loop
    print("start training...")

    for epoch in range(args.n_epochs):
        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0

        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]

            emb = extract_embed(node_embed, input_nodes)
            lbl = labels

            if th.cuda.is_available():
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()

            optimizer.zero_grad()
            if 0:
                # the following is the original code. Keep it for reference
                category = "paper"
                logits = model(emb, blocks)[category]
                # logits = model(emb, blocks)

                y_hat = logits.log_softmax(dim=-1)
                # y_hat = th.concat([logits[category][seeds[category]] for category in logits if category in seeds]).log_softmax(dim=-1)
                loss = F.nll_loss(y_hat, lbl)
            else:
                logits = model(emb, blocks)
                # logits = model(emb, blocks)
                loss = None
                for category in logits:
                    if category in seeds:
                        y_hat = logits[category].log_softmax(dim=-1)
                        if loss is None:
                            loss = F.nll_loss(y_hat, lbl[seeds[category]])
                        else:
                            loss += F.nll_loss(y_hat, lbl[seeds[category]])

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * args.batch_size
            # pbar.update(batch_size)

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, "
            f"Loss (w/o dividing sample num): {loss:.4f}, "
        )

    return  # logger


# TODO: implement logging to json and run all datasets
# TODO: Use conditional arguments to get a clearer structure of arguments as explained in https://stackoverflow.com/questions/9505898/conditional-command-line-arguments-in-python-using-argparse
def add_generic_RGNN_args(
    parser: argparse.ArgumentParser,
    default_logfilename: str,
    filtered_args: set[str] = {},
):
    if len(filtered_args) > 0:
        print(
            (
                "WARNING: add_generic_RGNN_args is called with these following"
                " removed arguments: "
            ),
            filtered_args,
        )
    parser.add_argument("--jit_script_enabled", action="store_true")
    parser.add_argument(
        "--logfile_enabled", action="store_true", help="enable logging to json"
    )
    parser.add_argument("--logfilename", type=str, default=default_logfilename)
    # DGL
    if not "dataset" in filtered_args:
        parser.add_argument(
            "-d", "--dataset", type=str, default="mag", help="dataset"
        )
    if not "n_infeat" in filtered_args:
        parser.add_argument(
            "--n_infeat",
            type=int,
            default=64,
            help=(
                "number of feature inputted into RGAT layer, which will be the"
                " output size of embedding layer when the latter is used"
            ),
        )
    if not "sparse_format" in filtered_args:
        parser.add_argument(
            "--sparse_format",
            type=str,
            default="csr",
            help="sparse format to use",
        )
    if not "sort_by_src" in filtered_args:
        parser.add_argument(
            "--sort_by_src", action="store_true", help="sort by src"
        )
    if not "sort_by_etype" in filtered_args:
        parser.add_argument(
            "--sort_by_etype", action="store_true", help="sort by etype"
        )
    if not "no_reindex_eid" in filtered_args:
        parser.add_argument(
            "--no_reindex_eid",
            action="store_true",
            help="use new eid after sorting rather than load referential eids",
        )
    if not "num_classes" in filtered_args:
        parser.add_argument(
            "--num_classes", type=int, default=8, help="number of classes"
        )
    if not "use_real_labels_and_features" in filtered_args:
        parser.add_argument(
            "--use_real_labels_and_features",
            action="store_true",
            help="use real labels",
        )
    if not "compact_direct_indexing_flag" in filtered_args:
        parser.add_argument(
            "--compact_direct_indexing_flag",
            action="store_true",
            default=False,
        )
    if not "lr" in filtered_args:
        parser.add_argument(
            "--lr", type=float, default=0.01, help="learning rate"
        )
    if not "num_heads" in filtered_args:
        parser.add_argument(
            "--num_heads", type=int, default=1, help="number of heads"
        )
    if not "n_epochs" in filtered_args:
        parser.add_argument(
            "-e",
            "--n_epochs",
            type=int,
            default=10,
            help="number of training epochs",
        )
    if not "fanout" in filtered_args:
        parser.add_argument("--fanout", type=int, nargs="+", default=[25, 20])
    if not "batch_size" in filtered_args:
        parser.add_argument("--batch_size", type=int, default=1024)
    if not "full_graph_training" in filtered_args:
        parser.add_argument("--full_graph_training", action="store_true")
    if not "num_layers" in filtered_args:
        parser.add_argument("--num_layers", type=int, default=1)
    if not "compact_as_of_node_flag" in filtered_args:
        parser.add_argument("--compact_as_of_node_flag", action="store_true")
    if not "no_warm_up" in filtered_args:
        parser.add_argument("--no_warm_up", action="store_true")
    # OGB
    if not "runs" in filtered_args:
        parser.add_argument("--runs", type=int, default=1)
    if not "dropout" in filtered_args:
        parser.add_argument(
            "--dropout", type=float, default=0.5, help="dropout probability"
        )
