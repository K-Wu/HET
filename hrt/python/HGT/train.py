#!/usr/bin/env python3
from .train_dgl import HGT_main_procedure, HGT_parse_args
from ..utils_lite import GRAPHILER_HETERO_DATASET

if __name__ == "__main__":
    args = HGT_get_and_parse_args()
    print(args)
    if args.dataset == "all":
        for dataset in GRAPHILER_HETERO_DATASET:
            args.dataset = dataset
            print(f"Training on {dataset}")
            HGT_main_procedure(args, dgl_model_flag=False)
    else:
        HGT_main_procedure(args, dgl_model_flag=False)
