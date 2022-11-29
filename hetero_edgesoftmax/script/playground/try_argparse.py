#!/usr/bin/env python3
import argparse


def my_add_argument(parser):
    parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
    parser.add_argument(
        "--take_in_list_test",
        nargs="+",
        type=int,
        help="take in list test",
        default=[10, 25],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    my_add_argument(parser)
    args = parser.parse_args()
    print(args)
    print(vars(args))
    print(args.sort_by_src)
    print(vars(args)["sort_by_src"])
