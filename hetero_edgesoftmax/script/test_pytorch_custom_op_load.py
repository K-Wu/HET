import torch

torch.ops.load_library("hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
print(torch.ops.torch_hetero_edgesoftmax.tensor_info)
print(torch.ops.torch_hetero_edgesoftmax.biops_tensor_info)
