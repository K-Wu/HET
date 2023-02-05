#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")
A = torch.tensor([[1.0, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
B = torch.tensor([[1.0, 2, 3, 4, 5, 6], [6, 7, 8, 9, 10, 11]])
C = torch.zeros(A.size()[0], B.size()[1])  # Any tensor as a place holder


def test_pass_argument(A, B, C):
    torch.ops.torch_hetero_edgesoftmax.rectangular_MatMul(A, B, C)
    torch.ops.torch_hetero_edgesoftmax.printTensor(C)


if __name__ == "__main__":
    # torch.ops.torch_hetero_edgesoftmax.build_debug_info()
    # torch.ops.torch_hetero_edgesoftmax.try_get_schedule_by_relations(100, 100)
    # print(torch.ops.torch_hetero_edgesoftmax.tensor_info)
    print(torch.ops.torch_hetero_edgesoftmax.rectangular_MatMul)
    print(torch.ops.torch_hetero_edgesoftmax.printTensor)
    # print(torch.ops.torch_hetero_edgesoftmax.biops_tensor_info)
    test_pass_argument(A, B, C)
