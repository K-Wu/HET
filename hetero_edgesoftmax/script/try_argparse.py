import argparse

parser = argparse.ArgumentParser(description="RGCN")
parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
args = parser.parse_args()
print(args)
