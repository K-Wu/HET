#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np

# import networkx as nx
import time
import torch
import torch.nn.functional as F

# from dgl import DGLGraph
# from dgl.data import register_data_args  # , load_data

# from dgl import transform
from .egl_gat import EglGAT, EglGATSingleLayer
from .GAT_utils import EarlyStopping
from .. import utils


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

    def seastar_original_load_dataset(args):
        path = "./dataset/" + str(args.dataset) + "/"
        """
        edges = np.loadtxt(path + 'edges.txt')
        edges = edges.astype(int)

        features = np.loadtxt(path + 'features.txt')

        train_mask = np.loadtxt(path + 'train_mask.txt')
        train_mask = train_mask.astype(int)

        labels = np.loadtxt(path + 'labels.txt')
        labels = labels.astype(int)
        """
        edges = np.load(path + "edges.npy")
        features = np.load(path + "features.npy")
        train_mask = np.load(path + "train_mask.npy")
        labels = np.load(path + "labels.npy")

        num_edges = edges.shape[0]
        num_nodes = features.shape[0]
        num_feats = features.shape[1]
        n_classes = int(max(labels) - min(labels) + 1)

        return (
            num_edges,
            num_nodes,
            num_feats,
            n_classes,
            edges,
            features,
            train_mask,
            labels,
        )


def GAT_train(args, single_layer_flag: bool):
    # load and preprocess dataset
    mydgl_graph, canonical_etype_indices_tuples = utils.RGNN_get_mydgl_graph(
        args.dataset,
        args.sort_by_src,
        args.sort_by_etype,
        args.no_reindex_eid,
        args.sparse_format,
    )
    num_nodes = mydgl_graph.get_num_nodes()
    num_edges = mydgl_graph.get_num_edges()

    # assert train_mask.shape[0] == num_nodes

    print("dataset {}".format(args.dataset))
    print("# of edges : {}".format(num_edges))
    print("# of nodes : {}".format(num_nodes))
    print("# of features : {}".format(args.num_feats))
    features = torch.randn(num_nodes, args.num_feats, requires_grad=True)
    labels = torch.from_numpy(np.random.randint(0, args.num_classes, num_nodes))
    train_idx = torch.randint(0, num_nodes, (args.train_size,))

    print(
        "WARNING: the original seastar loading dataset features is replaced with the seastar randomized features and labels instead."
    )
    print(
        "WARNING the original seastar train_mask utility is replaced with the seastar randomized train_mask instead."
    )

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        mydgl_graph.to(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_idx = train_idx.cuda()

    # initialize a DGL graph
    # TODO: incorporate the following modification to mydgl_graph
    # create model
    if not single_layer_flag:
        heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    if single_layer_flag:
        model = EglGATSingleLayer(
            mydgl_graph,
            args.num_feats,
            args.num_classes,
            args.num_heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual,
        )

    else:
        model = EglGAT(
            mydgl_graph,
            args.num_layers,
            args.num_feats,
            args.num_hidden,
            args.num_classes,
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.negative_slope,
            args.residual,
        )
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # initialize graph
    dur = []
    record_time = 0
    avg_run_time = 0
    Used_memory = 0

    for epoch in range(args.num_epochs):
        torch.cuda.synchronize()
        tf = time.time()
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)

        loss = loss_fcn(logits[train_idx], labels[train_idx])
        now_mem = torch.cuda.max_memory_allocated(0)
        print("now_mem : ", now_mem)
        Used_memory = max(now_mem, Used_memory)
        tf1 = time.time()

        optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        run_time_this_epoch = t2 - tf

        if epoch >= 3:
            dur.append(time.time() - t0)
            record_time += 1

            avg_run_time += run_time_this_epoch

        train_acc = accuracy(logits[train_idx], labels[train_idx])

        # log for each step
        print(
            "Epoch {:05d} | Time(s) {:.4f} | train_acc {:.6f} | Used_Memory {:.6f} mb".format(
                epoch, run_time_this_epoch, train_acc, (now_mem * 1.0 / (1024**2))
            )
        )
        """
        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):   
                    break

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc /{:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))
        
        """

    if args.early_stop:
        model.load_state_dict(torch.load("es_checkpoint.pt"))

    # OUTPUT we need
    avg_run_time = avg_run_time * 1.0 / record_time
    Used_memory /= 1024**3
    print("^^^{:6f}^^^{:6f}".format(Used_memory, avg_run_time))


def GAT_get_parser(single_layer_flag: bool):
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument(
        "--gpu", type=int, default=0, help="which GPU to use. Set -1 to use CPU."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="number of hidden attention heads"
    )
    parser.add_argument("--num_feats", type=int, default=128, help="input feature dim")

    if not single_layer_flag:
        parser.add_argument(
            "--num_out_heads",
            type=int,
            default=1,
            help="number of output attention heads",
        )
        parser.add_argument(
            "--num_layers", type=int, default=1, help="number of hidden layers"
        )
        parser.add_argument(
            "--num_hidden", type=int, default=32, help="number of hidden units"
        )
    parser.add_argument(
        "--residual", action="store_true", default=False, help="use residual connection"
    )
    parser.add_argument(
        "--in_drop", type=float, default=0.6, help="input feature dropout"
    )
    parser.add_argument(
        "--attn_drop", type=float, default=0.6, help="attention dropout"
    )
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        default=False,
        help="indicates whether to use early stop or not",
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="skip re-evaluate the validation set",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )
    parser.add_argument(
        "--sparse_format", type=str, default="csr", help="sparse format to use"
    )
    parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
    parser.add_argument("--sort_by_etype", action="store_true", help="sort by etype")
    parser.add_argument(
        "--no_reindex_eid",
        action="store_true",
        help="use new eid after sorting rather than load referential eids",
    )
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=4)
    return parser


if __name__ == "__main__":
    parser = GAT_get_parser(single_layer_flag=False)

    args = parser.parse_args()

    print(args)

    GAT_train(args, single_layer_flag=False)
