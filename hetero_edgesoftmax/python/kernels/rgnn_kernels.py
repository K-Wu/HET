#!/usr/bin/env python3
import torch


def rgnn_relational_matmul(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul(*args)


def rgnn_relational_matmul_backward(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul_backward(*args)


def rgnn_relational_matmul_ac_gather_scatter_list_identical(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul_ac_gather_scatter_list_identical(
        *args
    )


def rgnn_relational_matmul_backward_ac_gather_scatter_list_identical(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul_backward_ac_gather_scatter_list_identical(
        *args
    )


def rgnn_relational_matmul_compact_as_of_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul_compact_as_of_node(
        *args
    )


def backward_rgnn_relational_matmul_compact_as_of_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.backward_rgnn_relational_matmul_compact_as_of_node(
        *args
    )
