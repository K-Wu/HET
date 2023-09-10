#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py

# from pathlib import Path
from ..utils_lite import *
from . import mydglgraph_converters
import torch


def graphiler_load_data_as_mydgl_graph(
    name, to_homo=True, dataset_originally_homo_flag=False
):
    if to_homo:
        g, ntype_offsets, canonical_etype_indices_tuples = graphiler_load_data(
            name, to_homo=True
        )
        my_g = (
            mydglgraph_converters.create_mydgl_graph_coo_from_homo_dgl_graph(
                g, dataset_originally_homo_flag
            )
        )
        my_g["original"]["node_type_offsets"] = torch.LongTensor(ntype_offsets)
    else:
        g, ntype_offsets, canonical_etype_indices_tuples = graphiler_load_data(
            name, to_homo=False
        )  # feat_dim,
        my_g = (
            mydglgraph_converters.create_mydgl_graph_coo_from_hetero_dgl_graph(
                g
            )
        )
        my_g["original"]["node_type_offsets"] = torch.LongTensor(ntype_offsets)
    return my_g, ntype_offsets, canonical_etype_indices_tuples


def graphiler_setup_device(device="cuda:0"):
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    device = torch.device(device)
    return device


if __name__ == "__main__":
    # a place for testing data loading
    for dataset in GRAPHILER_HOMO_DATASET:
        graphiler_load_data(dataset)
    for dataset in GRAPHILER_HETERO_DATASET:
        graphiler_load_data(dataset)
