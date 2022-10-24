from ..utils import graphiler_datasets

OGB_DATASETS = ["arxiv", "proteins", "mag", "wikikg2", "biokg"]
DGL_DATASETS = ["reddit", "ppi", "cora", "pubmed", "aifb", "mutag", "bgs", "am"]
# len(g.etypes)
# "reddit" 1, "ppi" 1, "cora" 1, "pubmed" 1, "aifb" 104 (unique edge type names 78 canonical edge types 104)
# "arxiv"1, "proteins"1
# TODO: "pubmed" may trigger error: Edge type "ontology#name" is ambiguous. Please use canonical edge type in the form of (srctype, etype, dsttype)


def test_load_all_and_print_num_etypes():
    for datasets in [DGL_DATASETS, OGB_DATASETS]:
        for name in datasets:
            print(name)
            dataset = graphiler_datasets.graphiler_load_data_as_mydgl_graph(
                name, feat_dim=graphiler_datasets.GRAPHILER_DEFAULT_DIM, to_homo=False
            )


def test_load_aifb():
    name = "aifb"
    dataset = graphiler_datasets.graphiler_load_data_as_mydgl_graph(
        name, feat_dim=graphiler_datasets.GRAPHILER_DEFAULT_DIM
    )


if __name__ == "__main__":
    test_load_aifb()
