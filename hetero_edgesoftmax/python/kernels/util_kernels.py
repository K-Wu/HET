#!/usr/bin/env python3
import torch


def transpose_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.transpose_csr(*args)
