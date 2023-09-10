#!/usr/bin/env python3
from .train_dgl import RGAT_main_procedure, RGAT_parse_args
from ..utils_lite import GRAPHILER_HETERO_DATASET

if __name__ == "__main__":
    args = RGAT_parse_args()
    print(args)

    print(
        "WARNING: ignoring the hard-coded paper features in the original"
        " dataset in the original RGAT training script. This script is solely"
        " for performance R&D purposes."
    )
    if args.dataset == "all":
        for dataset in GRAPHILER_HETERO_DATASET:
            args.dataset = dataset
            print(f"Training on {dataset}")
            RGAT_main_procedure(args, dgl_model_flag=False)
    else:
        RGAT_main_procedure(args, dgl_model_flag=False)
