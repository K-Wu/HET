#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
from .rgcn_kernels import *
from .hgt_kernels import *
