#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hrt/libtorch_hrt.so")
torch.ops.torch_hrt.build_debug_info()
K = torch.ops.torch_hrt
