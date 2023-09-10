#!/usr/bin/env python3
from . import create_RGCN_parser, RGCNSingleLayer_main
from .. import utils_lite

# import torch as th
# import torch.nn.functional as F

if __name__ == "__main__":
    parser = create_RGCN_parser(RGCN_single_layer_flag=True)
    args = parser.parse_args()
    args.sparse_format = "separate_coo"
    print(args)
    args.bfs_level = 1 + 1  # num_layers + 1 pruning used nodes for memory

    if args.dataset == "all":
        for dataset in utils_lite.GRAPHILER_HETERO_DATASET:
            args.dataset = dataset
            print(f"Training on {dataset}")
            RGCNSingleLayer_main(args)
    else:
        RGCNSingleLayer_main(args)
