#!/usr/bin/env python3
from . import models
from . import models_dgl
import sys
import nvtx
import torch
from .. import utils
from .. import utils_lite

print(
    "WARNING: setting up device via"
    ' utils.graphiler_setup_device(device="cuda:0")'
)
device = utils_lite.graphiler_setup_device(device="cuda:0")


def profile(dataset, feat_dim, repeat=1000):
    print("benchmarking on: " + dataset)
    g, _, _2 = utils_lite.graphiler_load_data(dataset, feat_dim)
    g_hetero, _, _2 = utils_lite.graphiler_load_data(
        dataset, feat_dim, to_homo=False
    )
    # features = features.to(device)

    @utils_lite.empty_cache
    def run(g_hetero, features):
        with nvtx.annotate("dgl-slice", color="purple"):
            g_hetero = g_hetero.to(device)
            node_dict = {}
            edge_dict = {}
            for ntype in g_hetero.ntypes:
                node_dict[ntype] = len(node_dict)
            for etype in g_hetero.canonical_etypes:
                edge_dict[etype] = len(edge_dict)
            net_hetero = models_dgl.HGT_DGLHetero(
                node_dict,
                edge_dict,
                feat_dim,
                utils_lite.GRAPHILER_DEFAULT_DIM,
                utils_lite.GRAPHILER_DEFAULT_DIM,
            ).to(device)
            net_hetero.eval()
            with torch.no_grad():
                utils_lite.graphiler_bench(
                    net=net_hetero,
                    net_params=(g_hetero, g_hetero.ndata["h"]),
                    tag="1-DGL-slice",
                    nvprof=False,
                    repeat=repeat,
                    memory=True,
                )
            del g_hetero, node_dict, edge_dict, net_hetero


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python HGT.py [dataset] [feat_dim]")
        exit()
    if sys.argv[1] == "all":
        # log = {}
        for d in utils_lite.GRAPHILER_HETERO_DATASET:
            profile(d, utils_lite.GRAPHILER_DEFAULT_DIM, repeat=1000)
        # pd.DataFrame(log).to_pickle("output/HGT.pkl")
    elif sys.argv[1] == "breakdown":
        print("ERROR: ablation study not implemented yet!")
        exit(1)
    else:
        profile(sys.argv[1], int(sys.argv[2]), repeat=1000)
