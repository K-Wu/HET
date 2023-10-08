#!/usr/bin/env python3
import torch

try:
    torch.ops.load_library("../build/hrt/libtorch_hrt.so")
except Exception as ex:
    import traceback

    print("".join(traceback.TracebackException.from_exception(ex).format()))
    print(
        "This may suggest you are using a different environment than the one"
        " you built the library in."
    )
    exit(1)
torch.ops.torch_hrt.build_debug_info()
K = torch.ops.torch_hrt
