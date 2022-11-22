import torch

torch.ops.load_library("../build/hetero_edgesoftmax/libtorch_hetero_edgesoftmax.so")


def test_pass_argument(*args):
    torch.ops.torch_hetero_edgesoftmax.tensor_info(*args)


if __name__ == "__main__":
    torch.ops.torch_hetero_edgesoftmax.build_debug_info()
    torch.ops.torch_hetero_edgesoftmax.try_get_schedule_by_relations(100, 100)
    print(torch.ops.torch_hetero_edgesoftmax.tensor_info)
    print(torch.ops.torch_hetero_edgesoftmax.biops_tensor_info)
    test_pass_argument(torch.tensor([1, 2, 3, 4, 5]))
