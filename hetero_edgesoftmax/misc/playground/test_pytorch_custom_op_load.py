#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")


def test_pass_argument(*args):
    torch.ops.torch_hetero_edgesoftmax.tensor_info(*args)


def test_pass_biops_argument(*args):
    torch.ops.torch_hetero_edgesoftmax.biops_tensor_info(*args)


if __name__ == "__main__":
    torch.ops.torch_hetero_edgesoftmax.build_debug_info()
    torch.ops.torch_hetero_edgesoftmax.try_get_schedule_by_relations(100, 100)
    print(torch.ops.torch_hetero_edgesoftmax.tensor_info)
    print(torch.ops.torch_hetero_edgesoftmax.biops_tensor_info)
    test_pass_argument(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))
    test_tensor = torch.tensor([1, 2, 3, 4, 5])
    test_pass_biops_argument(test_tensor, test_tensor)
    tensor_dict = dict()
    for idx in range(10):
        tensor_dict[str(idx)] = torch.tensor([1, 2, 3, 4, 5])
    torch.ops.torch_hetero_edgesoftmax.print_tensor_dict_info(tensor_dict)
