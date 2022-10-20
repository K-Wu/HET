#!/usr/bin/env python3
import torch


def rgcn_layer0(graph, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0(graph, weight, norm, ret)


def rgcn_layer0_backward(graph, gradout, norm, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_backward(
        graph, gradout, norm, grad_weight
    )


def rgcn_layer1(graph, x, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1(graph, x, weight, norm, ret)


def rgcn_layer1_backward(graph, x, weight, norm, gradout, grad_x, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward(
        graph, x, weight, norm, gradout, grad_x, grad_weight
    )


def rgcn_layer0_csr(row_ptr, col_idx, eids, reltypes, weight, norm, ret):
    # print("row_ptr", row_ptr.dtype, row_ptr.device)
    # print("col_idx", col_idx.dtype, col_idx.device)
    # print("eids", eids.dtype, eids.device)
    # print("reltypes", reltypes.dtype, reltypes.device)
    # print("weight", weight.dtype, weight.device)
    # print("norm", norm.dtype, norm.device)
    # print("ret", ret.dtype, ret.device)
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_csr(
        row_ptr, col_idx, eids, reltypes, weight, norm, ret
    )


def rgcn_layer0_backward_csr(
    row_ptr, col_idx, eids, reltypes, gradout, norm, grad_weight
):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_backward_csr(
        row_ptr, col_idx, eids, reltypes, gradout, norm, grad_weight
    )


def rgcn_layer1_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_csr(
        row_ptr, col_idx, eids, reltypes, x, weight, norm, ret
    )


def rgcn_layer1_backward_csr(
    row_ptr, col_idx, eids, reltypes, x, weight, norm, gradout, grad_x, grad_weight
):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward_csr(
        row_ptr, col_idx, eids, reltypes, x, weight, norm, gradout, grad_x, grad_weight
    )
