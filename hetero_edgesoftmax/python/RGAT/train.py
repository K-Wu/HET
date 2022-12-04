#!/usr/bin/env python3
from .train_dgl import RGAT_main_procedure, RGAT_parse_args

if __name__ == "__main__":
    args = RGAT_parse_args()
    print(args)

    print(
        "WARNING: ignoring the hard-coded paper features in the original dataset in the original RGAT training script. This script is solely for performance R&D purposes."
    )
    RGAT_main_procedure(args, dgl_model_flag=False)
