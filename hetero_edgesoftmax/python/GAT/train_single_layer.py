#!/usr/bin/env python3
from .train import GAT_get_parser, GAT_train

if __name__ == "__main__":
    parser = GAT_get_parser(single_layer_flag=True)
    args = parser.parse_args()

    print(args)
    GAT_train(args, single_layer_flag=True)
