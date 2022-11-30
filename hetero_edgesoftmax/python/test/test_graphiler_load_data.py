#!/usr/bin/env python3
from ..utils import graphiler_datasets

OGB_DATASETS = ["arxiv", "proteins", "mag", "wikikg2", "biokg"]
DGL_DATASETS = ["reddit", "ppi", "cora", "pubmed", "aifb", "mutag", "bgs", "am"]
# len(g.etypes)
# "reddit" 1, "ppi" 1, "cora" 1, "pubmed" 1, "aifb" 104 (unique edge type names 78 canonical edge types 104), "mutag" 50 (unique edge type names 46), "bgs"
# "arxiv"1, "proteins"1
# TODO: "pubmed" may trigger error: Edge type "ontology#name" is ambiguous. Please use canonical edge type in the form of (srctype, etype, dsttype)

# Datasets where node index in (src, dest) are starting from zero for each unique node/canoncial type: "aifb". "mutag"
# Datasets that keep absolute node index: None if ignoring graph with only one edge type/ node type
def test_load_dataset(name):
    print(name)
    dataset = graphiler_datasets.graphiler_load_data_as_mydgl_graph(name)


def test_load_all_and_print_num_etypes():
    for datasets in [DGL_DATASETS, OGB_DATASETS]:
        for name in datasets:
            test_load_dataset(name)


if __name__ == "__main__":
    test_load_dataset("bgs")
