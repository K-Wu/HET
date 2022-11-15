import argparse

parser = argparse.ArgumentParser(description="RGCN")
parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
parser.add_argument(
    "--take_in_list_test",
    nargs="+",
    type=int,
    help="take in list test",
    default=[10, 25],
)
args = parser.parse_args()
print(args)
print(vars(args))
print(args.sort_by_src)
