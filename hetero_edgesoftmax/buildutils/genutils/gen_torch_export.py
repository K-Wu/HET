# this file generates the op.cu.cc that involves all the torch kernels logic and their export statements.
import os
import sys

file_beginning = """
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <iostream>
#include <map>

#include "DGLHackKernel/DGLHackKernel.h"
#include "hetero_edgesoftmax.h"
"""
generated_files = ["DebugInfo.inc.h"]
generated_files = list(map(lambda x: "generated/" + x, generated_files))


# usage python gen_torch_export.py ${PROJECT_SOURCE_DIR}
if __name__ == "__main__":
    # use sys.argv[1] as the path to the root of the project
    if len(sys.argv) != 2:
        print("Usage: python gen_torch_export.py ${PROJECT_SOURCE_DIR}")
        exit(1)
    root_path = sys.argv[1]

    # get all the .inc.h files in DGLHackKernel/OpExport/
    inc_files = []
    for root, dirs, files in os.walk(
        os.path.join(
            root_path, "hetero_edgesoftmax", "include", "DGLHackKernel", "OpExport"
        )
    ):
        for file in files:
            if file.endswith(".inc.h"):
                inc_files.append(file)
    inc_files = list(map(lambda x: "DGLHackKernel/OpExport/" + x, inc_files))

    # print all the content
    print(file_beginning)

    for file in inc_files:
        print('#include "' + file + '"')

    for file in generated_files:
        print('#include "' + file + '"')
