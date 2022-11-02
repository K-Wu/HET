#!/usr/bin/env python3
import torch


def fused_gat_kernel_csr(*args):
    torch.ops.torch_hetero_edgesoftmax.fused_gat_kernel_csr(*args)


def backward_fused_gat_csr(*args):
    torch.ops.torch_hetero_edgesoftmax.backward_fused_gat_csr(*args)
