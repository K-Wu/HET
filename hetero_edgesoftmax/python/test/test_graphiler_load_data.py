from ..utils import graphiler_datasets

OGB_DATASETS = ["arxiv", "proteins", "mag", "wikikg2", "biokg"]
DGL_DATASETS = ["reddit", "ppi", "cora", "pubmed", "aifb", "mutag", "bgs", "am"]

if __name__ == "__main__":
    for datasets in [DGL_DATASETS, OGB_DATASETS]:
        for name in datasets:
            print(name)
            dataset = graphiler_datasets.graphiler_load_data_as_mydgl_graph(
                name, feat_dim=graphiler_datasets.GRAPHILER_DEFAULT_DIM, to_homo=False
            )
