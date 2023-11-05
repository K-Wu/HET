#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F

import nvtx

# from dgl import DGLGraph
# from dgl.contrib.data import load_data
from .. import backend as B
from .. import utils
from .. import utils_lite
from ..RGNNUtils import *

from typing import Union


# from functools import partial


"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn


class Seastar_EglRelGraphConv(nn.Module):
    @utils_lite.warn_default_arguments
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_edges,
        sparse_format,
        regularizer="basis",
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
        layer_type=0,
        hybrid_assign_flag=False,
        num_blocks_on_node_forward=-1,
        num_blocks_on_node_backward=-1,
        compact_as_of_node_flag=False,
        compact_direct_indexing_flag=False,
    ):
        # TODO: pass num_heads if applicable to RGCN
        super(Seastar_EglRelGraphConv, self).__init__()
        self.hybrid_assign_flag = hybrid_assign_flag
        self.num_blocks_on_node_forward = num_blocks_on_node_forward
        self.num_blocks_on_node_backward = num_blocks_on_node_backward
        self.compact_as_of_node_flag = compact_as_of_node_flag
        if self.compact_as_of_node_flag:
            raise NotImplementedError(
                "compact_as_of_node_flag not implemented yet"
            )
        self.compact_direct_indexing_flag = compact_direct_indexing_flag
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if (
            self.num_bases is None
            or self.num_bases > self.num_rels
            or self.num_bases <= 0
        ):
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.layer_type = layer_type
        self.sparse_format = sparse_format

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(
                th.Tensor(self.num_bases, self.in_feat, self.out_feat)
            )
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(
                    th.Tensor(self.num_rels, self.num_bases)
                )
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain("relu")
            )
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(
                    self.w_comp, gain=nn.init.calculate_gain("relu")
                )
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, norm=None):
        """Forward computation

        Parameters
        ----------
        g : HET_DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :mathtorch.bmm(A.unsqueeze(0).expand_as(v), v):`(|E|, 1)`

        Returns
        -------
        torch.Tensor
            New node features.
        """

        # TODO: pass num_heads if applicable to RGCN
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(
                self.num_bases, self.in_feat * self.out_feat
            )
            print(
                "weight size:",
                self.weight.size(),
                "w_comp size:",
                self.w_comp.size(),
            )
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight

        if self.sparse_format == "csr":
            if self.layer_type == 0:
                if self.hybrid_assign_flag:
                    raise NotImplementedError
                node_repr = B.seastar_rgcn_layer0_csr(
                    g, weight, norm
                )  # NB: this line uses my own rgcn_layer0
            else:
                node_repr = B.seastar_rgcn_layer1_csr(
                    g,
                    x,
                    weight,
                    norm,
                    self.hybrid_assign_flag,
                    self.num_blocks_on_node_forward,
                    self.num_blocks_on_node_backward,
                )  # NB: this line uses my own rgcn_layer1
        else:
            assert self.sparse_format == "coo"
            if self.layer_type == 0:
                raise NotImplementedError(
                    "Only support csr format for layer 0"
                )
                node_repr = B.rgcn_layer0_coo(g, weight, norm)
            else:
                if self.hybrid_assign_flag:
                    raise NotImplementedError
                node_repr = B.seastar_rgcn_layer1_coo(g, x, weight, norm)
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr


class HET_EglRelGraphConv_EdgeParallel(nn.Module):
    """Based on Seastar_EglRelGraphConv"""

    @utils_lite.warn_default_arguments
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_edges,
        regularizer="basis",
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=False,
        compact_as_of_node_flag=False,
        compact_direct_indexing_flag=False,
        dropout=0.0,
        layer_type=0,
    ):
        # TODO: pass num_heads if applicable to RGCN
        super(HET_EglRelGraphConv_EdgeParallel, self).__init__()
        if self_loop:
            raise NotImplementedError
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.compact_direct_indexing_flag = compact_direct_indexing_flag
        self.num_bases = num_bases
        if (
            self.num_bases is None
            or self.num_bases > self.num_rels
            or self.num_bases <= 0
        ):
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.layer_type = layer_type
        self.compact_direct_indexing_flag = compact_direct_indexing_flag

        if regularizer == "basis":
            # add basis weights

            self.weight = nn.Parameter(
                th.Tensor(self.num_bases, self.in_feat, self.out_feat)
            )
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(
                    th.Tensor(self.num_rels, self.num_bases)
                )
            nn.init.xavier_uniform_(
                self.weight, gain=nn.init.calculate_gain("relu")
            )
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(
                    self.w_comp, gain=nn.init.calculate_gain("relu")
                )
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, norm=None):
        """Forward computation

        Parameters
        ----------
        g : HET_DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :mathtorch.bmm(A.unsqueeze(0).expand_as(v), v):`(|E|, 1)`

        Returns
        -------
        torch.Tensor
            New node features.
        """

        # TODO: pass num_heads if applicable to RGCN
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(
                self.num_bases, self.in_feat * self.out_feat
            )
            print(
                "weight size:",
                self.weight.size(),
                "w_comp size:",
                self.w_comp.size(),
            )
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
        else:
            weight = self.weight

        if self.compact_as_of_node_flag:
            weight = weight.unsqueeze(1)

        if self.layer_type == 0:
            raise NotImplementedError
        else:
            # TODO: add support to compact_direct_indexing_flag
            if self.compact_as_of_node_flag:
                separate_unique_node_indices_single_sided = (
                    g.get_separate_unique_node_indices_single_sided()
                )
                feat_compact_src = B.rgnn_relational_matmul(
                    {
                        "unique_srcs_and_dests_rel_ptrs": separate_unique_node_indices_single_sided[
                            "rel_ptrs_row"
                        ],
                        "unique_srcs_and_dests_node_indices": separate_unique_node_indices_single_sided[
                            "node_indices_row"
                        ],
                    },
                    weight,
                    x,
                    True,
                    1,  # CompactAsOfNodeKind::Enabled
                )
                feat_compact_src = feat_compact_src.squeeze(1)
                node_repr = B.rgcn_node_mean_aggregation_compact_as_of_node_separate_coo_single_sided(
                    g,
                    feat_compact_src,
                    norm,
                    self.compact_direct_indexing_flag,
                )  # NB: use single side instead without need to modify kernel
            else:
                node_repr = B.rgcn_layer1_separate_coo(
                    g,
                    x,
                    weight,
                    norm,
                )  # NB: this line uses my own rgcn_layer1

        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr


class HET_EGLRGCNSingleLayerModel(nn.Module):
    @utils_lite.warn_default_arguments
    def __init__(
        self,
        mydglgraph,
        n_infeat,
        out_dim,
        num_rels,
        num_nodes,
        num_edges,
        sparse_format,
        num_bases,
        dropout,
        activation,
        hybrid_assign_flag,
        num_blocks_on_node_forward,
        num_blocks_on_node_backward,
        compact_as_of_node_flag,
        compact_direct_indexing_flag,
    ):
        self.mydglgraph = mydglgraph
        # TODO: pass num_heads if applicable to RGCN
        self.num_nodes = num_nodes
        super(HET_EGLRGCNSingleLayerModel, self).__init__()
        if sparse_format == "separate_coo":
            self.layer2 = HET_EglRelGraphConv_EdgeParallel(
                n_infeat,
                out_dim,
                num_rels,
                num_edges,
                num_bases=num_bases,
                dropout=dropout,
                activation=activation,
                layer_type=1,
                compact_as_of_node_flag=compact_as_of_node_flag,
                compact_direct_indexing_flag=compact_direct_indexing_flag,
            )
        else:
            self.layer2 = Seastar_EglRelGraphConv(
                n_infeat,
                out_dim,
                num_rels,
                num_edges,
                sparse_format,
                num_bases=num_bases,
                dropout=dropout,
                activation=activation,
                layer_type=1,
                compact_as_of_node_flag=compact_as_of_node_flag,
                compact_direct_indexing_flag=compact_direct_indexing_flag,
            )

    def forward(self, feats, edge_norm):
        h = self.layer2.forward(self.mydglgraph, feats, edge_norm)
        return h


class Seastar_EGLRGCNModel(nn.Module):
    @utils_lite.warn_default_arguments
    def __init__(
        self,
        mydglgraph,
        num_nodes,
        hidden_dim,
        out_dim,
        num_rels,
        num_edges,
        sparse_format,
        num_bases,
        dropout,
        activation,
        compact_direct_indexing_flag,
    ):
        self.mydglgraph = mydglgraph
        self.num_nodes = num_nodes
        super(Seastar_EGLRGCNModel, self).__init__()
        self.layer1 = Seastar_EglRelGraphConv(
            num_nodes,
            hidden_dim,
            num_rels,
            num_edges,
            sparse_format,
            num_bases=num_bases,
            dropout=dropout,
            activation=activation,
            compact_direct_indexing_flag=compact_direct_indexing_flag,
            layer_type=0,
        )
        self.layer2 = Seastar_EglRelGraphConv(
            hidden_dim,
            out_dim,
            num_rels,
            num_edges,
            sparse_format,
            num_bases=num_bases,
            dropout=dropout,
            activation=activation,
            compact_direct_indexing_flag=compact_direct_indexing_flag,
            layer_type=1,
        )

    def forward(self, feats, edge_norm):
        h = self.layer1.forward(self.mydglgraph, feats, edge_norm)
        h = self.layer2.forward(self.mydglgraph, h, edge_norm)
        return h


def get_seastar_model(args, mydglgraph):
    num_nodes = mydglgraph.get_num_nodes()
    num_rels = mydglgraph.get_num_rels()
    num_edges = mydglgraph.get_num_edges()
    num_classes = args.num_classes
    model = Seastar_EGLRGCNModel(
        mydglgraph,
        num_nodes,
        args.hidden_size,
        num_classes,
        num_rels,
        num_edges,
        args.sparse_format,
        num_bases=args.num_bases,
        activation=None,  # F.relu,
        compact_direct_indexing_flag=args.compact_direct_indexing_flag,
        dropout=args.dropout,
    )
    return model


def main_seastar(args):
    g, canonical_etype_indices_tuples = utils.RGNN_get_mydgl_graph(
        args.dataset,
        args.sort_by_src,
        args.sort_by_etype,
        args.no_reindex_eid,
        args.sparse_format,
    )

    g = g.cuda().contiguous()
    # Execute g = g.to_script_object() here so that 1) the script object veresion is stored as model.mydglgraph, and succeeding operation on g after get_our_model is applicable to model.mydglgraph
    # TODO: fix this in future if it breaks
    # g = g.to_script_object()

    # TODO: need to move g to cuda and then store g in the model
    model: Seastar_EGLRGCNModel = get_seastar_model(args, g)
    num_nodes = g.get_num_nodes()
    # since the nodes are featureless, the input feature is then the node id.

    # TODO: 3. use RelGraphEmbed so that we can do embedding lookup embed_layer = RelGraphEmbed(g, args.n_infeat, exclude=[])
    # embed_layer = RelGraphEmbed(
    #     g, args.n_infeat, exclude=[]
    # )
    node_embed_layer = HET_RelGraphEmbed(
        g, 1, dtype=torch.int64, requires_grad=False
    )
    # node_embed = node_embed_layer()
    # g = g.to_script_object()
    if not args.full_graph_training:
        raise NotImplementedError(
            "Not full_graph_training in"
            " RGAT_main_procedure(dgl_model_flag == False)"
        )
        # TODO: 1. Use convert_sampled_iteration_to_mydgl_graph from mydglgraph_converters to convert blocks[-1] to mydglgraph
        # TODO: 2. use embedding_lookup with the input node indices tensor returned by step 1 to aggregate the input_nodes features. e.g. node_embed = node_embed_layer() line in HET_RGNN_train_full_graph in RGNNUtils.py

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


def RGCN_prepare_data(args, g, model, node_embed):
    """This function creates the necessary tensors and move necessary tensors to cuda device. The logic in the function is originally at the beginning of RGCN_main_procedure(), but moved out to this individual function in an attempt to unify RGCN_main_procedure() with HET_RGNN_train_full_graph()."""
    labels = np.random.randint(0, args.num_classes, model.num_nodes)

    # edge type and normalization factor
    edge_norm = torch.rand(g["original"]["eids"].size())
    labels = torch.from_numpy(labels).view(-1).long()

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        # This operation is effective because reference to g is stored in model and this operation does to() in place, i.e., without creating new g, all tensors as values of g's dictionaries is replaced with new tensors on device, while the keys stay the same.

        torch.cuda.set_device(args.gpu)
        node_embed = node_embed.cuda()
        # print(node_embed.shape)
        # edge_type = g["original"]["rel_types"]
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()
        model = model.cuda()
    node_embed.requires_grad_(True)
    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm
    )

    return labels, edge_norm, node_embed, model, optimizer


def RGCN_eval_during_train(
    logits,
    loss,
    model: Union[Seastar_EGLRGCNModel, HET_EGLRGCNSingleLayerModel],
    node_embed_layer: HET_RelGraphEmbed,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
    edge_norm,
):
    # split dataset into train, validate, test
    train_idx = torch.randint(0, model.num_nodes, (args.train_size,))
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5 :]
    else:
        val_idx = train_idx
    train_idx = list(train_idx)
    # print(
    #     "Epoch {:05d} | Train Forward Time(s) {:.4f} (our kernel"
    #     " {:.4f} cross entropy {:.4f}) | Backward Time(s) {:.4f}".format(
    #         epoch,
    #         forward_time[-1],
    #         (tb - t0),
    #         (t1 - ta),
    #         backward_time[-1],
    #     )
    # )
    train_acc = torch.sum(
        logits[train_idx].argmax(dim=1) == labels[train_idx]
    ).item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc = torch.sum(
        logits[val_idx].argmax(dim=1) == labels[val_idx]
    ).item() / len(val_idx)
    print(
        "Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation"
        " Accuracy: {:.4f} | Validation loss: {:.4f}".format(
            train_acc, loss.item(), val_acc, val_loss.item()
        )
    )


def RGCN_test_after_train(
    model: Union[Seastar_EGLRGCNModel, HET_EGLRGCNSingleLayerModel],
    node_embed_layer: HET_RelGraphEmbed,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
    edge_norm,
):
    test_idx = torch.randint(0, model.num_nodes, (args.test_size,))
    # Evaluating the accuracy
    # TODO: move out the folloign code to an individual function
    model.eval()
    logits = model.forward(node_embed_layer(), edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    print(
        "Test Accuracy: {:.4f} | Test loss: {:.4f}".format(
            test_acc, test_loss.item()
        )
    )


def RGCN_main_procedure(
    model: Union[Seastar_EGLRGCNModel, HET_EGLRGCNSingleLayerModel],
    node_embed_layer: HET_RelGraphEmbed,
    optimizer,
    labels: th.Tensor,
    args: argparse.Namespace,
    edge_norm,
):
    # TODO: argument specify num_classes, len(train_idx), len(test_idx) for now.
    """
    aifb len(labels) == 8285, num_nodes == 8285, num_relations == 91, num_edges == 66371, len(train_idx) == 140, len(test_idx) == 36, num_classes = 4
    mutag len(labels) == 23644, num_nodes == 23644, num_relations == 47, num_edges == 172098, len(train_idx) == 272, len(test_idx) == 68, num_classes = 2
    bgs len(labels) == 333845, num_nodes == 333845, num_relations == 207, num_edges == 2166243, len(train_idx) == 117, len(test_idx) == 29, num_classes = 2
    """

    # TODO: Unify RGCN_main_procedure with HET_RGNN_train_full_graph.
    # TODO: stage 1) use HET_RGNN_train_full_graph, and forget about the args.verbose that get calculate accuracy during and after training for now
    # TODO: stage 2) add args.verbose functionality to HET_RGNN_train_full_graph

    if args.jit_script_enabled:
        model = torch.jit.script(model)

    forward_time = []
    backward_time = []
    training_time = []

    warm_up_forward_intermediate_memory = []
    warm_up_intermediate_memory = []

    model.train()

    memory_offset = torch.cuda.memory_allocated()
    # warm up
    if not args.no_warm_up:
        for epoch in range(5):
            optimizer.zero_grad()
            logits = model(node_embed_layer(), edge_norm)
            # forward_intermediate_memory needs to be collected before backward
            warm_up_forward_intermediate_memory.append(
                torch.cuda.max_memory_allocated() - memory_offset
            )
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            warm_up_intermediate_memory.append(
                torch.cuda.max_memory_allocated() - memory_offset
            )
            # Reset the memory stat to clear up the maximal memory allocated stat.
            reset_peak_memory_stats()

    # training loop
    print("start training...")
    for epoch in range(args.n_epochs):
        # with nvtx.annotate("training"):
        # with torch.cuda.profiler.profile():
        # nvtx.push_range("training", domain="my_domain")
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        logits = model(node_embed_layer(), edge_norm)
        torch.cuda.synchronize()
        tb = time.time()
        ta = time.time()
        loss = F.cross_entropy(logits, labels)
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        # nvtx.pop_range()
        # if epoch >= 3:
        forward_time.append(tb - t0)
        backward_time.append(t2 - t1)
        training_time.append(t2 - t0)

        print(
            f"Epoch: {epoch + 1 :02d}, "
            f"Loss (w/o dividing sample num): {loss.item():.4f}, "
        )
        print(f"Forward prop time: {forward_time[-1] * 1000} ms")
        print(f"Backward prop time: {backward_time[-1] * 1000} ms")
        if args.verbose:
            RGCN_eval_during_train(
                logits,
                loss,
                model,
                node_embed_layer,
                optimizer,
                labels,
                args,
                edge_norm,
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
                np.mean(forward_time[len(forward_time) // 4 :]) * 1000
            )
        )
        print(
            "Mean backward time: {:4f} ms".format(
                np.mean(backward_time[len(backward_time) // 4 :]) * 1000
            )
        )
        print(
            "Mean training time: {:4f} ms".format(
                np.mean(training_time[len(training_time) // 4 :]) * 1000
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

    if args.verbose:
        RGCN_test_after_train(model, node_embed_layer, optimizer, labels, args)


def create_RGCN_parser(RGCN_single_layer_flag):
    parser = argparse.ArgumentParser(description="RGCN")

    if RGCN_single_layer_flag:
        add_generic_RGNN_args(
            parser,
            "RGCN.json",
            {"use-self-loop", "runs", "use_real_labels_and_features"},
        )
    else:
        add_generic_RGNN_args(
            parser,
            "RGCN.json",
            {
                "use-self-loop",
                "runs",
                "use_real_labels_and_features",
                "n_infeat",
            },
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=16,
            help="number of hidden units",
        )
    print(
        "WARNING: RGCN currently does not support the following batch_size,"
        " !full_graph_training, num_heads"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--num_bases",
        type=int,
        default=-1,
        help="number of filter weight matrices, default: -1 [use all]",
    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument(
        "--relabel",
        default=False,
        action="store_true",
        help="remove untouched nodes and relabel",
    )
    parser.add_argument(
        "--use-self-loop",
        default=False,
        action="store_true",
        help="include self feature as a special relation",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="print out training information",
    )

    parser.add_argument("--hybrid_assign_flag", action="store_true")
    parser.add_argument(
        "--num_blocks_on_node_forward",
        type=int,
    )
    parser.add_argument(
        "--num_blocks_on_node_backward",
        type=int,
    )
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=64)
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    parser.set_defaults(validation=True)
    return parser


if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=False)
    args = parser.parse_args()
    print(args)
    args.bfs_level = args.num_layers + 1  # pruning used nodes for memory
    if args.dataset == "all":
        for dataset in utils_lite.GRAPHILER_HETERO_DATASET:
            args.dataset = dataset
            main_seastar(args)
    else:
        main_seastar(args)
