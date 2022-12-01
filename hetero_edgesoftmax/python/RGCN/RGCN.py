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

# from dgl import DGLGraph
# from dgl.contrib.data import load_data
from .. import backend as B
from .. import utils
from ..RGNNUtils import *

# from functools import partial


"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn


class HET_EglRelGraphConv(nn.Module):
    @utils.warn_default_arguments
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
    ):
        super(HET_EglRelGraphConv, self).__init__()
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
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))
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
        # print('aaa',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        # torch.cuda.synchronize()
        # t1 = time.time()
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases, self.in_feat * self.out_feat)
            # print('weight size:', self.weight.size(), 'w_comp size:', self.w_comp.size())
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat
            )
            # print('new weight size:', weight.size())
        else:
            weight = self.weight
        # torch.cuda.synchronize()
        # print('bbb',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        # t2 = time.time()

        if self.sparse_format == "csr":
            if self.layer_type == 0:
                node_repr = B.rgcn_layer0_csr(
                    g, weight, norm
                )  # NB: this line uses my own rgcn_layer0
                # print('output of layer 0', node_repr)
            else:
                node_repr = B.rgcn_layer1_csr(
                    g, x, weight, norm
                )  # NB: this line uses my own rgcn_layer1
        else:
            assert self.sparse_format == "coo"
            if self.layer_type == 0:
                raise NotImplementedError("Only support csr format for layer 0")
                node_repr = B.rgcn_layer0_coo(g, weight, norm)
                # print('output of layer 0', node_repr)
            else:
                node_repr = B.rgcn_layer1_coo(g, x, weight, norm)
        # torch.cuda.synchronize()
        # t3 = time.time()
        # print('output of layer 1', node_repr)
        # print('ccc',th.cuda.memory_allocated(),th.cuda.max_memory_allocated())
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        # torch.cuda.synchronize()
        # t4 = time.time()
        # print('matmul takes:',t2-t1, 's', (t2-t1)/(t4-t1),'%')
        # print('gcn takes:',t3-t2, 's', (t3-t2)/(t4-t1),'%')
        # print('rest takes:',t4-t3, 's', (t4-t3)/(t4-t1),'%')
        return node_repr


class HET_EGLRGCNSingleLayerModel(nn.Module):
    @utils.warn_default_arguments
    def __init__(
        self,
        n_infeat,
        out_dim,
        num_rels,
        num_edges,
        sparse_format,
        num_bases,
        dropout,
        activation,
    ):
        super(HET_EGLRGCNSingleLayerModel, self).__init__()
        self.layer2 = HET_EglRelGraphConv(
            n_infeat,
            out_dim,
            num_rels,
            num_edges,
            sparse_format,
            num_bases=num_bases,
            dropout=dropout,
            activation=activation,
            layer_type=1,
        )

    def forward(self, g, feats, edge_norm):
        h = self.layer2.forward(g, feats, edge_norm)
        return h


class HET_EGLRGCNModel(nn.Module):
    @utils.warn_default_arguments
    def __init__(
        self,
        num_nodes,
        hidden_dim,
        out_dim,
        num_rels,
        num_edges,
        sparse_format,
        num_bases,
        dropout,
        activation,
    ):
        super(HET_EGLRGCNModel, self).__init__()
        self.layer1 = HET_EglRelGraphConv(
            num_nodes,
            hidden_dim,
            num_rels,
            num_edges,
            sparse_format,
            num_bases=num_bases,
            dropout=dropout,
            activation=activation,
            layer_type=0,
        )
        self.layer2 = HET_EglRelGraphConv(
            hidden_dim,
            out_dim,
            num_rels,
            num_edges,
            sparse_format,
            num_bases=num_bases,
            dropout=dropout,
            activation=activation,
            layer_type=1,
        )

    def forward(self, g, feats, edge_norm):
        h = self.layer1.forward(g, feats, edge_norm)
        h = self.layer2.forward(g, h, edge_norm)
        return h


def get_model(args, mydglgraph):
    num_nodes = mydglgraph.get_num_nodes()
    num_rels = mydglgraph.get_num_rels()
    num_edges = mydglgraph.get_num_edges()
    num_classes = args.num_classes
    model = HET_EGLRGCNModel(
        num_nodes,
        args.hidden_size,
        num_classes,
        num_rels,
        num_edges,
        args.sparse_format,
        num_bases=args.num_bases,
        activation=F.relu,
        dropout=args.dropout,
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
    model = get_model(args, g)
    num_nodes = g["original"]["row_ptr"].numel() - 1
    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)

    RGCN_main_procedure(args, g, model, feats)


def RGCN_main_procedure(args, g, model, feats):
    # TODO: argument specify num_classes, len(train_idx), len(test_idx) for now.
    # aifb len(labels) == 8285, num_nodes == 8285, num_relations == 91, num_edges == 66371, len(train_idx) == 140, len(test_idx) == 36, num_classes = 4
    # mutag len(labels) == 23644, num_nodes == 23644, num_relations == 47, num_edges == 172098, len(train_idx) == 272, len(test_idx) == 68, num_classes = 2
    # bgs len(labels) == 333845, num_nodes == 333845, num_relations == 207, num_edges == 2166243, len(train_idx) == 117, len(test_idx) == 29, num_classes = 2
    # num_nodes = g["original"]["row_ptr"].numel() - 1
    if args.sparse_format == "coo":
        num_nodes = int(th.max(g["original"]["row_idx"]))
    else:
        assert args.sparse_format == "csr"
        num_nodes = g["original"]["row_ptr"].numel() - 1
    # num_rels = int(g["original"]["rel_types"].max().item()) + 1
    num_classes = args.num_classes
    labels = np.random.randint(0, num_classes, num_nodes)
    train_idx = torch.randint(0, num_nodes, (args.train_size,))
    test_idx = torch.randint(0, num_nodes, (args.test_size,))

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5 :]
    else:
        val_idx = train_idx

    # edge type and normalization factor
    edge_type = g["original"]["rel_types"]
    edge_norm = torch.rand(g["original"]["eids"].size())
    labels = torch.from_numpy(labels).view(-1).long()

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        g.cuda()
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm
    )

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    train_labels = labels[train_idx]
    train_idx = list(train_idx)
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.time()
        logits = model(g, feats, edge_norm)
        torch.cuda.synchronize()
        tb = time.time()
        train_logits = logits[train_idx]
        ta = time.time()
        # loss = F.cross_entropy(train_logits, train_labels)
        loss = F.cross_entropy(logits, labels)
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        if epoch >= 3:
            forward_time.append(tb - t0)
            backward_time.append(t2 - t1)
            if args.verbose:
                print(
                    "Epoch {:05d} | Train Forward Time(s) {:.4f} (our kernel {:.4f} cross entropy {:.4f}) | Backward Time(s) {:.4f}".format(
                        epoch, forward_time[-1], (tb - t0), (t1 - ta), backward_time[-1]
                    )
                )
        train_acc = torch.sum(
            logits[train_idx].argmax(dim=1) == labels[train_idx]
        ).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(
            logits[val_idx].argmax(dim=1) == labels[val_idx]
        ).item() / len(val_idx)
        if args.verbose:
            print(
                "Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
                    train_acc, loss.item(), val_acc, val_loss.item()
                )
            )
    print("max memory allocated", torch.cuda.max_memory_allocated())

    model.eval()
    logits = model.forward(g, feats, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(
        logits[test_idx].argmax(dim=1) == labels[test_idx]
    ).item() / len(test_idx)
    print(
        "Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item())
    )
    print()

    print(
        "Mean forward time: {:4f}".format(
            np.mean(forward_time[len(forward_time) // 4 :])
        )
    )
    print(
        "Mean backward time: {:4f}".format(
            np.mean(backward_time[len(backward_time) // 4 :])
        )
    )

    Used_memory = torch.cuda.max_memory_allocated(0) / (1024**3)
    avg_run_time = np.mean(forward_time[len(forward_time) // 4 :]) + np.mean(
        backward_time[len(backward_time) // 4 :]
    )
    # output we need
    print("^^^{:6f}^^^{:6f}".format(Used_memory, avg_run_time))


def _deprecated_create_RGCN_parser(RGCN_single_layer_flag: bool):
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    if RGCN_single_layer_flag:
        parser.add_argument(
            "--n_infeat", type=int, default=16, help="number of hidden units"
        )
    else:
        parser.add_argument(
            "--hidden_size", type=int, default=16, help="number of hidden units"
        )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--num_bases",
        type=int,
        default=-1,
        help="number of filter weight matrices, default: -1 [use all]",
    )
    if not RGCN_single_layer_flag:
        parser.add_argument(
            "--num_layers", type=int, default=2, help="number of propagation rounds"
        )
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--sparse_format",
        type=str,
        default="csr",
        help="sparse format",
        choices=["coo", "csr"],  # noqa: E501
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )
    parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
    parser.add_argument("--sort_by_etype", action="store_true", help="sort by etype")
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
    parser.add_argument(
        "--reindex_eid",
        action="store_true",
        help="use new eid after sorting rather than load referential eids",
    )
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--test_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=4)
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    parser.set_defaults(validation=True)
    return parser


def create_RGCN_parser(RGCN_single_layer_flag):
    parser = argparse.ArgumentParser(description="RGCN")
    if RGCN_single_layer_flag:
        add_generic_RGNN_args(
            parser, {"use-self-loop", "runs", "use_real_labels_and_features"}
        )
    else:
        add_generic_RGNN_args(
            parser,
            {
                "use-self-loop",
                "runs",
                "use_real_labels_and_features",
                "n_infeat",
            },
        )
        parser.add_argument(
            "--hidden_size", type=int, default=16, help="number of hidden units"
        )
    print(
        "WARNING: RGCN currently does not support the following batch_size, !full_graph_training, n_head"
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
    main(args)
