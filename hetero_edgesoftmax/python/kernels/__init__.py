#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
torch.ops.torch_hetero_edgesoftmax.build_debug_info()
K = torch.ops.torch_hetero_edgesoftmax
