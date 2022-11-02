#!/usr/bin/env python3
import torch


def backward_relational_fused_gat_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.backward_relational_fused_gat_csr(*args)


def relational_fused_gat_kernel_csr(*args):
    return torch.ops.torch_hetero_edgesoftmax.relational_fused_gat_kernel_csr(*args)


def rgat_relational_matmul(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgat_relational_matmul(*args)


def rgat_relational_matmul_backward(*args):
    return torch.ops.torch_hetero_edgesoftmax.rgat_relational_matmul_backward(*args)
