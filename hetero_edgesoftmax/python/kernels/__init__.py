#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
torch.ops.torch_hetero_edgesoftmax.build_debug_info()
K = torch.ops.torch_hetero_edgesoftmax
# from .rgcn_kernels import *
# from .hgt_kernels import *
# from .util_kernels import *
# from .rgat_kernels import *
# from .rgnn_kernels import *
# from .gat_kernels import *
