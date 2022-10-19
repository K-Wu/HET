#!/usr/bin/env python3
from . import models
import sys
import nvtx
import torch
from .. import utils


def profile(dataset, feat_dim, repeat=1000):
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim, prepare=False)
    g_hetero, _ = load_data(dataset, feat_dim, to_homo=False)
    features = features.to(device)

    @empty_cache
    def run(g_hetero, features):
        with nvtx.annotate("dgl-slice", color="purple"):
            g_hetero = g_hetero.to(device)
            node_dict = {}
            edge_dict = {}
            for ntype in g_hetero.ntypes:
                node_dict[ntype] = len(node_dict)
            for etype in g_hetero.canonical_etypes:
                edge_dict[etype] = len(edge_dict)
            net_hetero = models.HGT_DGLHetero(
                node_dict, edge_dict, feat_dim, DEFAULT_DIM, DEFAULT_DIM
            ).to(device)
            net_hetero.eval()
            with torch.no_grad():
                utils.bench(
                    net=net_hetero,
                    net_params=(g_hetero, g_hetero.ndata["h"]),
                    tag="1-DGL-slice",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                    log=log,
                )
            del g_hetero, node_dict, edge_dict, net_hetero


def breakdown(dataset, feat_dim, repeat=1000):
    # log = init_log(['0-DGL-UDF', '1+compile', '2+reorder',
    #               '3+fusion'], ['time', 'mem'])
    print("benchmarking on: " + dataset)
    g, features = load_data(dataset, feat_dim)
    g, features = g.to(device), features.to(device)
    net = HGT(feat_dim, DEFAULT_DIM, DEFAULT_DIM, g.num_ntypes, g.num_rels).to(device)
    net.eval()
    with torch.no_grad():
        utils.bench(
            net=net,
            net_params=(g, features, "naive"),
            tag="0-DGL-UDF",
            nvprof=False,
            repeat=repeat,
            memory=True,
            log=log,
        )
        global BREAK_FLAG
        BREAK_FLAG = 0
        utils.bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="1+compile",
            nvprof=False,
            repeat=repeat,
            memory=True,
        )  # , log=log)
        BREAK_FLAG = 1
        utils.bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="2+reorder",
            nvprof=False,
            repeat=repeat,
            memory=True,
        )  # , log=log)
        BREAK_FLAG = 2
        utils.bench(
            net=net,
            net_params=(g, features, "compile"),
            tag="3+fusion",
            nvprof=False,
            repeat=repeat,
            memory=True,
        )  # , log=log)

    return log


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python HGT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        # log = {}
        for d in hetero_dataset:
            profile(d, DEFAULT_DIM, repeat)
        # pd.DataFrame(log).to_pickle("output/HGT.pkl")
    elif sys.argv[1] == "breakdown":
        print("not migrated yet!")
        exit(1)
        log = {}
        for d in hetero_dataset:
            log[d] = breakdown(d, DEFAULT_DIM, repeat)
        # pd.DataFrame(log).to_pickle("output/HGT_breakdown.pkl")
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat)
