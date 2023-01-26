#!/usr/bin/env python3
from .. import utils
from ..utils_lite import GRAPHILER_HETERO_DATASET

OGB_DATASETS = ["arxiv", "proteins", "mag", "wikikg2", "biokg"]
DGL_DATASETS = [
    "reddit",
    "ppi",
    "cora",
    "pubmed",
    "aifb",
    "mutag",
    "bgs",
    "am",
    "fb15k",
]
# len(g.etypes)
# "reddit" 1, "ppi" 1, "cora" 1, "pubmed" 1, "aifb" 104 (unique edge type names 78 canonical edge types 104), "mutag" 50 (unique edge type names 46), "bgs"
# "arxiv"1, "proteins"1
# TODO: "pubmed" may trigger error: Edge type "ontology#name" is ambiguous. Please use canonical edge type in the form of (srctype, etype, dsttype)

# Datasets where node index in (src, dest) are starting from zero for each unique node/canoncial type: "aifb". "mutag"
# Datasets that keep absolute node index: None if ignoring graph with only one edge type/ node type
def test_load_dataset(name, dataset_originally_homo_flag):
    print(name)
    (
        dataset,
        ntype_offsets,
        canonical_etype_idx_tuples,
    ) = utils.graphiler_load_data_as_mydgl_graph(
        name, to_homo=True, dataset_originally_homo_flag=dataset_originally_homo_flag
    )
    scripted_dataset = dataset.to_script_object()
    assert dataset.get_num_ntypes() == len(ntype_offsets) - 1
    print(
        dataset.get_num_ntypes(),
        dataset.get_num_rels(),
        dataset.get_num_nodes(),
        dataset.get_num_edges(),
    )
    print(canonical_etype_idx_tuples)
    pass


def test_load_all_and_print_num_etypes():
    for dataset in DGL_DATASETS + OGB_DATASETS:
        test_load_dataset(dataset, not dataset in GRAPHILER_HETERO_DATASET)


if __name__ == "__main__":
    print(locals())
    print(globals())
    # test_load_dataset("bgs")
    # test_load_dataset("fb15k", dataset_originally_homo_flag=False)
    test_load_all_and_print_num_etypes()
