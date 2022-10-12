import torch
torch.ops.load_library("build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
print(torch.ops.torch_hetero_edgesoftmax.dummy_warp_perspective)