#!/usr/bin/env python3
import torch


# def hgt_full_graph_hetero_message_ops_csr(
#     row_ptr, col_idx, eids, reltypes, weight, applied_vlinear_node_features, ret
# ):
#     return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_message_ops_csr(
#         row_ptr, col_idx, eids, reltypes, weight, applied_vlinear_node_features, ret
#     )


# def hgt_full_graph_hetero_message_ops_backward_csr(
#     row_ptr,
#     col_idx,
#     eids,
#     reltypes,
#     weight,
#     applied_vlinear_node_features,
#     gradout,
#     grad_weight,
#     grad_v,
# ):
#     return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_message_ops_backward_csr(
#         row_ptr,
#         col_idx,
#         eids,
#         reltypes,
#         weight,
#         applied_vlinear_node_features,
#         gradout,
#         grad_weight,
#         grad_v,
#     )


def hgt_full_graph_hetero_attention_ops_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_attention_ops_csr(
        *args
    )


def hgt_full_graph_hetero_attention_ops_backward_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_hetero_attention_ops_backward_csr(
        *args
    )


def hgt_full_graph_message_mean_aggregation_backward_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_message_mean_aggregation_backward_csr(
        *args
    )


def hgt_full_graph_message_mean_aggregation_csr(*args):
    return (
        torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_message_mean_aggregation_csr(
            *args
        )
    )


def hgt_full_graph_edge_softmax_ops_backward_csr(*args):
    return (
        torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_edge_softmax_ops_backward_csr(
            *args
        )
    )


def hgt_full_graph_edge_softmax_ops_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.hgt_full_graph_edge_softmax_ops_csr(*args)
