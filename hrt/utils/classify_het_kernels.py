from .detect_pwd import get_het_root_path
from .nsight_utils import (
    is_ctags_installed,
    classify_fw_bw_kernel,
    get_functions_from_ctags_table,
)
from typing import Tuple
import os


def get_fw_bw_host_func_names(het_root_path: str) -> Tuple[set[str], set[str]]:
    # the kernel is a forward kernel if its namespace involve "FwProp"
    # the kernel is a backward kernel if its namespace involve "BckProp"
    assert is_ctags_installed(), "ctags is not installed"
    torch_op_files = [
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/HGTOpsEdgeParallel.inc.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/RGCNOps.inc.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/GATOps.inc.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/HGTOps.inc.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/RGATOps.inc.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/OpExport/RGNNOps.inc.h",
    ]

    # Use exuberant ctags to get the function names.
    ctags_tables: list[str] = []
    for file in torch_op_files:
        assert os.path.isfile(file), f"{file} does not exist"
        ctags_tables.append(os.popen("ctags -f- " + file).read())

    fw_kernels = set()
    bw_kernels = set()
    for table in ctags_tables:
        for line in table.split("\n"):
            line_components = line.split("\t")
            if len(line_components) < 3:
                print(
                    f"WARNING: ctags table line {line} has less than 3"
                    " components"
                )
                continue
            func_name = line_components[0]
            namespace = line_components[-1]
            tag_type = line_components[-2]
            if tag_type != "f":
                print(f"Warning: {func_name} is not a function")
                continue
            if "FwProp" in namespace:
                fw_kernels.add(func_name)
            elif "BckProp" in namespace:
                bw_kernels.add(func_name)
            else:
                print(
                    f"Warning: {func_name} is neither a forward nor a backward"
                    " kernel"
                )

    return fw_kernels, bw_kernels


def get_GEMM_kernel_names(het_root_path: str) -> "set[str]":
    assert is_ctags_installed(), "ctags is not installed"
    # HET_ functions defined in the two files are GEMM kernels.
    GEMM_files = [
        f"{het_root_path}/hrt/include/DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h",
        f"{het_root_path}/hrt/include/DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h",
    ]

    # Use exuberant ctags to get the function names.
    ctags_tables: list[str] = []
    for file in GEMM_files:
        assert os.path.isfile(file), f"{file} does not exist"
        ctags_tables.append(os.popen("ctags -f- " + file).read())

    return get_functions_from_ctags_table("\n".join(ctags_tables))


GEMM_kernels: "set[str]" = get_GEMM_kernel_names(get_het_root_path())


def test_classify_fw_bw_kernel():
    kernel_names = set()
    with open(
        f"{get_het_root_path()}/hrt/utils/test/kernel_names_trace.test_log"
    ) as f:
        for line in f:
            kernel_names.add(line.strip())

    for func_pretty_name in kernel_names:
        if classify_fw_bw_kernel(func_pretty_name) == "FwProp":
            continue
        elif classify_fw_bw_kernel(func_pretty_name) == "BckProp":
            continue
        else:
            print(
                f"{func_pretty_name} is neither a forward nor a backward"
                " kernel"
            )
            assert 0


def classify_het_kernel(func_name: str) -> str:
    if func_name in GEMM_kernels:
        return "GEMM"
    elif func_name.startswith("HET_"):
        return "Traversal"
    else:
        return "Non-HET Others"


# TODO: load nsys report and output the time portion of the three types of kernels.
# cuda_gpu_kern_sum.py


if __name__ == "__main__":
    print(is_ctags_installed())
    print(get_GEMM_kernel_names(get_het_root_path()))
    print(get_fw_bw_host_func_names(get_het_root_path()))
    test_classify_fw_bw_kernel()
