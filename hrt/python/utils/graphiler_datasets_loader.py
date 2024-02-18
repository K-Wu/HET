#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py

# from pathlib import Path
from __future__ import annotations
from typing import Any, Callable
from ..utils_lite import *
from . import mydglgraph_converters
from . import MyDGLGraph
from . import graph_data_key_to_function
import torch
from dgl.base import EID, ETYPE, NID, NTYPE
import numpy as np


def graphiler_load_data_as_mydgl_graph(
    name, to_homo=True, dataset_originally_homo_flag=False
) -> tuple[MyDGLGraph, list[int], list[tuple[int, int, int]]]:
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


def is_multi_level_key_in(
    key: tuple[str, ...], graph_data: dict[Any, Any]
) -> bool:
    curr_dict: dict[Any, Any] | Any = graph_data
    for level_key in key:
        if level_key not in curr_dict:
            return False
        curr_dict = curr_dict[level_key]
    return True


def get_funcs_to_propagate_and_produce_metadata(
    graph: MyDGLGraph,
) -> list[Callable]:
    funcs = list()
    if "legacy_metadata_from_dgl" in graph.graph_data:
        funcs.append(
            lambda my_g: my_g.graph_data.update(
                {
                    "legacy_metadata_from_dgl": graph.graph_data[
                        "legacy_metadata_from_dgl"
                    ]
                }
            )
        )

    for key, func_name in graph_data_key_to_function:
        if is_multi_level_key_in(key, graph.graph_data):
            print("Found key:", key)
            # Capture func_name as func_name_
            # See https://docs.python.org/3.4/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
            funcs.append(
                lambda my_g, func_name_=func_name: getattr(my_g, func_name_)()
            )
    return funcs


if __name__ == "__main__":
    # a simple routine that tests data loading

    for dataset in GRAPHILER_HETERO_DATASET:
        print(f"Now working on {dataset}")
        g_hetero, ntype_offsets, _2 = graphiler_load_data(
            dataset, to_homo=False
        )
        g = dgl.to_homogeneous(g_hetero)
        ntypes = g_hetero.ntypes
        etypes = g_hetero.etypes
        canonical_etypes = g_hetero.canonical_etypes
        print("g num nodes", g.number_of_nodes())
        print("g edges type", g.edata[ETYPE])
        print("g nodes type", g.ndata[NTYPE])
        print("g.ntypes", g.ntypes)
        print("g.etypes", g.etypes)
        num_nodes = ntype_offsets[-1]
        # Only one level of neighbour (1 layer)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            list(range(num_nodes)),
            sampler,
            batch_size=1024,
            shuffle=True,
            drop_last=False,
            num_workers=4,
        )
        # print(g.edata)
        # print(g.ndata)
        # Reference: https://docs.dgl.ai/en/1.1.x/generated/dgl.dataloading.NeighborSampler.html#dgl.dataloading.NeighborSampler
        # DGL Dataloader inherits pytorch's torch.utils.data.DataLoader https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py

        for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if idx == 0:
                # input_nodes[i] and output_nodes[i] maps subgraph node index, `i`, to the node index in the original graph. block.edges() stores the node pair with subgraph node index.
                print(input_nodes, output_nodes, blocks)
                last_block = dgl.block_to_graph(blocks[-1])
                print(last_block.etypes)
                print(last_block.ntypes)
                print(last_block.ndata[NTYPE])
                print(type(last_block.ntypes))

                last_block = dgl.reorder_graph(
                    last_block,
                    edge_permute_algo="custom",
                    node_permute_algo="custom",
                    permute_config={
                        "nodes_perm": np.argsort(blocks[-1].ndata[NID]),
                        "edges_perm": np.argsort(blocks[-1].edata[EID]),
                    },
                )

                print("block[-1] etype", blocks[-1].edata[ETYPE])
                print("block[-1] ntype", blocks[-1].ndata[NTYPE])
                print("last_block etype", last_block.edata[ETYPE])
                print("last_block ntype", last_block.ndata[NTYPE])
                print(blocks[-1].edges())

                print(max(blocks[-1].edges()[0]))
                print(max(blocks[-1].edges()[1]))

                # Use convert_sampled_iteration_to_mydgl_graph from mydglgraph_converters to convert blocks[-1] to mydglgraph
                (
                    my_g,
                    canonical_etypes_id_tuple,
                ) = mydglgraph_converters.convert_sampled_iteration_to_mydgl_graph(
                    blocks
                )

    for dataset in GRAPHILER_HOMO_DATASET:
        g, _, _2 = graphiler_load_data(dataset)
