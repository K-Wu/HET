#!/usr/bin/env python3
import torch


def rgat_layer_backward_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgat_layer_backward_csr(*args)


def rgat_layer_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgat_layer_csr(*args)
