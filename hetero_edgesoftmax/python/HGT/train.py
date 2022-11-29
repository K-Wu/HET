#!/usr/bin/env python3
from .train_dgl import HGT_main_procedure, HGT_parse_args

if __name__ == "__main__":
    args = HGT_parse_args()
    print(args)
    HGT_main_procedure(args, dgl_model_flag=False)
