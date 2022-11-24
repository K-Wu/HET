#!/usr/bin/env python3
import torch

# TODO: to reduce redundancy, we may program to establish function definitions dynamically from a list of function name strings,
# During run time, the program works on the function name in the list one by one, define the function whose name is exactly the string in the list, and whose body is invocation of the same-name method in torch.ops.torch_hetero_edgesoftmax.
# To get a function pointer by string name, we may use getattr() e.g. getattr(torch.ops.torch_hetero_edgesoftmax,'tensor_info')(torch.rand(15))
# To define a function by a string name, we may use https://stackoverflow.com/questions/55915109/how-to-define-a-function-name-using-a-string


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


def rgnn_inner_product_node_compact_and_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_inner_product_node_compact_and_node(
        *args
    )


def backward_rgnn_inner_product_node_compact_and_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.backward_rgnn_inner_product_node_compact_and_node(
        *args
    )


def rgnn_inner_product_edge_and_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_inner_product_edge_and_node(*args)


def backward_rgnn_inner_product_edge_and_node(*args):
    return torch.ops.torch_hetero_edgesoftmax.backward_rgnn_inner_product_edge_and_node(
        *args
    )


def rgnn_relational_matmul_compact_as_of_node_single_ended(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgnn_relational_matmul_compact_as_of_node_single_ended(
        *args
    )


def backward_rgnn_relational_matmul_compact_as_of_node_single_ended(*args):
    return torch.ops.torch_hetero_edgesoftmax.backward_rgnn_relational_matmul_compact_as_of_node_single_ended(
        *args
    )
