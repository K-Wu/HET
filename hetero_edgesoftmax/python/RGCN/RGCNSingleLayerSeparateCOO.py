#!/usr/bin/env python3
from . import create_RGCN_parser, RGCNSingleLayer_main
from .. import utils
import torch as th
import torch.nn.functional as F

if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=True)
    args = parser.parse_args()
    args.sparse_format = "separate_coo"
    print(args)
    args.bfs_level = 1 + 1  # num_layers + 1 pruning used nodes for memory
    RGCNSingleLayer_main(args)
