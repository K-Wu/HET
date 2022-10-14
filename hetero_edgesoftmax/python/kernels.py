import torch
torch.ops.load_library("build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")

def rgcn_layer0(graph, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0(graph, weight, norm, ret)

def rgcn_layer0_backward(graph, gradout, norm, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_backward(graph, gradout, norm, grad_weight)

def rgcn_layer1(graph, x, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1(graph, x, weight, norm, ret)

def rgcn_layer1_backward(graph, x, weight, norm, gradout, grad_x, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward(graph, x, weight, norm, gradout, grad_x, grad_weight)

def rgcn_layer0_csr(row_ptr, col_idx, eids, reltypes, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_csr(row_ptr, col_idx, eids, reltypes, weight, norm, ret)

def rgcn_layer0_backward_csr(row_ptr, col_idx, eids, reltypes, gradout, norm, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer0_backward_csr(row_ptr, col_idx, eids, reltypes, gradout, norm, grad_weight)

def rgcn_layer1_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, ret):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, ret)


def rgcn_layer1_backward_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, gradout, grad_x, grad_weight):
    return torch.ops.torch_hetero_edgesoftmax.rgcn_layer1_backward_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, gradout, grad_x, grad_weight)