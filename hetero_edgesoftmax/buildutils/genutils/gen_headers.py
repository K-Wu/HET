#!/usr/bin/env python3
import argparse


def print_debug_info_func():
    debug_info_func = """
    #pragma once

    #include <c10/cuda/CUDAException.h>
    #include <c10/cuda/CUDAStream.h>
    #include <torch/extension.h>
    #include <torch/library.h>
    #include "DGLHackKernel/RGNN/mysgemm_KernelsBlockConfigurations.h"


    void build_debug_info() {
    std::cout << "GIT_COMMIT_HASH: " << GIT_COMMIT_HASH << std::endl;
    std::cout << "built for CUDA ARCHS " << CUDA_ARCHS << std::endl;
    #ifdef ENABLE_DEBUG_MACRO
    std::cout << "WARNING: library built in debug mode without -O3" << std::endl;
    #else
    std::cout << "library built in release mode with -O3" << std::endl;
    #endif
    std::cout << "library compiled by gcc " << __GNUC__ << "." << __GNUC_MINOR__
                << "." << __GNUC_PATCHLEVEL__ << ", nvcc " << __CUDACC_VER_MAJOR__
                << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__
                << std::endl;
    }

    TORCH_LIBRARY_FRAGMENT(torch_hetero_edgesoftmax, m) {
        m.def("build_debug_info", build_debug_info);
        }
    """
    print(debug_info_func)


if __name__ == "__main__":
    # if the arg is --gen_debug_info, then generate the debug info function
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_debug_info", action="store_true")
    args = parser.parse_args()
    if args.gen_debug_info:
        print_debug_info_func()
