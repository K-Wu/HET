#!/usr/bin/env python3
# adapted from graphiler/python/graphiler/utils/setup.py

# from pathlib import Path
from ..utils_lite import *
from . import mydglgraph_converters
from . import MyDGLGraph
import torch


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


def yield_batch_as_mydglgraph(
    dataloader: dgl.dataloading.DataLoader,
    funcs_to_apply: set[str] = {
        "generate_separate_unique_node_indices_for_each_etype",
        "generate_separate_unique_node_indices_single_sided_for_each_etype",
    },
):
    for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        (
            input_nodes,
            output_nodes,
            my_g,
            canonical_etypes_id_tuple,
        ) = mydglgraph_converters.convert_sampled_iteration_to_mydgl_graph(
            input_nodes, output_nodes, blocks
        )

        for func_name in funcs_to_apply:
            getattr(my_g, func_name)()

        yield my_g


if __name__ == "__main__":
    # a place for testing data loading

    for dataset in GRAPHILER_HETERO_DATASET:
        print(f"Now working on {dataset}")
        g, ntype_offsets, _2 = graphiler_load_data(dataset)
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
        print(g.edata)
        print(g.ndata)
        # Reference: https://docs.dgl.ai/en/1.1.x/generated/dgl.dataloading.NeighborSampler.html#dgl.dataloading.NeighborSampler
        # DGL Dataloader inherits pytorch's torch.utils.data.DataLoader https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py

        for idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if idx == 0:
                # input_nodes[i] and output_nodes[i] maps subgraph node index, `i`, to the node index in the original graph. block.edges() stores the node pair with subgraph node index.
                print(input_nodes, output_nodes, blocks)
                print(blocks[-1].edges())
                print(dgl.to_homogeneous(blocks[-1], return_count=True))
                print(max(blocks[-1].edges()[0]))
                print(max(blocks[-1].edges()[1]))

                # Use convert_sampled_iteration_to_mydgl_graph from mydglgraph_converters to convert blocks[-1] to mydglgraph
                (
                    input_nodes,
                    output_nodes,
                    my_g,
                    canonical_etypes_id_tuple,
                ) = mydglgraph_converters.convert_sampled_iteration_to_mydgl_graph(
                    input_nodes, output_nodes, blocks
                )

    for dataset in GRAPHILER_HOMO_DATASET:
        g, _, _2 = graphiler_load_data(dataset)
