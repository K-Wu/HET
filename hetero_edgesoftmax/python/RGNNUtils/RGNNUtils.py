#!/usr/bin/env python3
from typing import Union
import torch as th
import torch.nn.functional as F
from torch import nn
from .. import utils

from dgl.heterograph import DGLBlock


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

    @utils.warn_default_arguments
    def __init__(self, g, embed_size, exclude=list()):

        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size

        # create learnable embeddings for all nodes, except those with a node-type in the "exclude" list
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if ntype in exclude:
                continue
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            self.embeds[ntype] = embed

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeds.values():
            nn.init.xavier_uniform_(emb)

    def forward(self, block: Union[None, DGLBlock] = None):
        return self.embeds


class HET_RelGraphEmbed(nn.Module):
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

    @utils.warn_default_arguments
    def __init__(self, g: utils.MyDGLGraph, embed_size, exclude=list()):

        super(HET_RelGraphEmbed, self).__init__()
        self.embed_size = embed_size

        # create learnable embeddings for all nodes, except those with a node-type in the "exclude" list

        self.embeds = nn.Parameter(th.Tensor(g.get_num_nodes(), self.embed_size))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embeds)

    def forward(self, block: Union[None, DGLBlock] = None):
        return self.embeds


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb


def HET_RGNN_train_with_sampler(
    g, model, node_embed, optimizer, train_loader, labels, device, hypermeters
):
    raise NotImplementedError("HET_RGNN_train_with_sampler not implemented yet")


def HET_RGNN_train_full_graph(
    g: utils.MyDGLGraph,
    model,
    node_embed_layer,
    optimizer,
    labels: th.Tensor,
    device,
    hypermeters: dict,
):
    # training loop
    print("start training...")
    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()
        model.requires_grad_(True)
        node_embed_layer.train()
        node_embed_layer.requires_grad_(True)

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset in the original RGAT training script. This script is solely for performance R&D purposes."
        )

        # emb = extract_embed(node_embed, input_nodes)
        # emb = node_embed

        # Add the batch's raw "paper" features
        # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})
        node_embed = node_embed_layer()

        optimizer.zero_grad()
        forward_prop_start = th.cuda.Event(enable_timing=True)
        forward_prop_end = th.cuda.Event(enable_timing=True)
        forward_prop_start.record()
        logits = model(g, node_embed)
        # logits = model(emb, blocks)

        y_hat = logits.log_softmax(dim=-1)
        loss = F.nll_loss(y_hat, labels)
        forward_prop_end.record()
        th.cuda.synchronize()

        backward_prop_start = th.cuda.Event(enable_timing=True)
        backward_prop_end = th.cuda.Event(enable_timing=True)
        backward_prop_start.record()

        loss.backward()
        optimizer.step()
        backward_prop_end.record()
        th.cuda.synchronize()

        # FIXME: should be # edges when training full graph
        total_loss += loss.item() * hypermeters["batch_size"]

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
        print(
            f"Forward prop time: {forward_prop_start.elapsed_time(forward_prop_end)} ms"
        )
        print(
            f"Backward prop time: {backward_prop_start.elapsed_time(backward_prop_end)} ms"
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGNN_train_full_graph(model, node_embed, optimizer, labels, hypermeters: dict):
    # training loop
    print("start training...")
    category = "paper"
    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )

        # emb = extract_embed(node_embed, input_nodes)
        emb = node_embed

        # Add the batch's raw "paper" features
        # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})

        lbl = labels

        if th.cuda.is_available():
            emb = {k: e.cuda() for k, e in emb.items()}
            lbl = {k: e.cuda() for k, e in lbl.items()}

        optimizer.zero_grad()
        forward_prop_start = th.cuda.Event(enable_timing=True)
        forward_prop_end = th.cuda.Event(enable_timing=True)
        forward_prop_start.record()
        logits = model(emb)
        # logits = model(emb, blocks)
        loss = None
        for category in logits:
            y_hat = logits[category].log_softmax(dim=-1)
            if loss is None:
                loss = F.nll_loss(y_hat, lbl)
            else:
                loss += F.nll_loss(y_hat, lbl)
        forward_prop_end.record()
        th.cuda.synchronize()

        backward_prop_start = th.cuda.Event(enable_timing=True)
        backward_prop_end = th.cuda.Event(enable_timing=True)
        backward_prop_start.record()
        loss.backward()
        optimizer.step()
        backward_prop_end.record()
        th.cuda.synchronize()

        # FIXME: should be # edges when training full graph
        total_loss += loss.item() * hypermeters["batch_size"]

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
        print(
            f"Forward prop time: {forward_prop_start.elapsed_time(forward_prop_end)} ms"
        )
        print(
            f"Backward prop time: {backward_prop_start.elapsed_time(backward_prop_end)} ms"
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGNN_train_with_sampler(
    model, node_embed, optimizer, train_loader, labels, device, hypermeters: dict
):
    # training loop
    print("start training...")

    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )
        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]
            # seeds = seeds[category]  # we only predict the nodes with type "category"

            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features
            # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})

            # lbl = th.concat([labels[seeds[category]] for category in seeds])
            # lbl = labels[seeds]
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

            total_loss += loss.item() * hypermeters["batch_size"]
            # pbar.update(batch_size)

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


# TODO: Use conditional arguments to get a clearer structure of arguments as explained in https://stackoverflow.com/questions/9505898/conditional-command-line-arguments-in-python-using-argparse
def add_generic_RGNN_args(parser):
    # DGL
    parser.add_argument("-d", "--dataset", type=str, default="ogbn-mag", help="dataset")
    parser.add_argument(
        "--n_infeat",
        type=int,
        default=64,
        help="number of feature inputted into RGAT layer, which will be the output size of embedding layer when the latter is used",
    )
    parser.add_argument(
        "--sparse_format", type=str, default="csr", help="sparse format to use"
    )
    parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
    parser.add_argument("--sort_by_etype", action="store_true", help="sort by etype")
    parser.add_argument(
        "--reindex_eid",
        action="store_true",
        help="use new eid after sorting rather than load referential eids",
    )
    parser.add_argument("--num_classes", type=int, default=8, help="number of classes")
    parser.add_argument(
        "--use_real_labels_and_features", action="store_true", help="use real labels"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n_head", type=int, default=2, help="number of heads")
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=3, help="number of training epochs"
    )
    parser.add_argument("--fanout", type=int, nargs="+", default=[25, 20])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--full_graph_training", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--compact_as_of_node_flag", action="store_true")
    # OGB
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )