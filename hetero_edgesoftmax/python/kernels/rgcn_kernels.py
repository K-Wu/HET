#!/usr/bin/env python3
import torch


def rgcn_layer0_csr(*args):
    # print("row_ptr", row_ptr.dtype, row_ptr.device)
    # print("col_idx", col_idx.dtype, col_idx.device)
    # print("eids", eids.dtype, eids.device)
    # print("reltypes", reltypes.dtype, reltypes.device)
    # print("weight", weight.dtype, weight.device)
    # print("norm", norm.dtype, norm.device)
    # print("ret", ret.dtype, ret.device)
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_csr(*args)


def rgcn_layer0_backward_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_backward_csr(*args)


def rgcn_layer1_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_csr(*args)


def rgcn_layer1_backward_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward_csr(*args)


def rgcn_layer1_coo(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_coo(*args)


def rgcn_layer1_backward_coo(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward_coo(*args)
