#!/usr/bin/env python3
import torch

torch.ops.load_library("../build/hrt/libtorch_hrt.so")


def try_pass_argument(*args):
    torch.ops.torch_hrt.tensor_info(*args)


def try_pass_biops_argument(*args):
    torch.ops.torch_hrt.biops_tensor_info(*args)


if __name__ == "__main__":
    torch.ops.torch_hrt.build_debug_info()
    torch.ops.torch_hrt.try_get_schedule_by_relations(100, 100)
    print(torch.ops.torch_hrt.tensor_info)
    print(torch.ops.torch_hrt.biops_tensor_info)
    try_pass_argument(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32))
    try_tensor = torch.tensor([1, 2, 3, 4, 5])
    try_pass_biops_argument(try_tensor, try_tensor)
    tensor_dict = dict()
    for idx in range(10):
        tensor_dict[str(idx)] = torch.tensor([1, 2, 3, 4, 5])
    torch.ops.torch_hrt.print_tensor_dict_info(tensor_dict)
