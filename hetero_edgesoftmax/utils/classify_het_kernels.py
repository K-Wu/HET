from .detect_pwd import get_het_root_path, run_once
from functools import lru_cache
import os


@lru_cache(maxsize=None)
@run_once
def is_ctags_installed() -> bool:
    return os.system("ctags --version >/dev/null 2>/dev/null") == 0


def get_functions_from_ctags_table(ctags_table: str) -> set[str]:
    result = set()
    for line in ctags_table.split("\n"):
        if line.startswith("HET_") and line.endswith("f"):
            result.add(line.split("\t")[0])
    return result


def get_GEMM_kernel_names() -> "set[str]":
    assert is_ctags_installed(), "ctags is not installed"
    # HET_ functions defined in the two files are GEMM kernels.
    GEMM_files = [
        f"{get_het_root_path()}/hetero_edgesoftmax/include/DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h",
        f"{get_het_root_path()}/hetero_edgesoftmax/include/DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h",
    ]

    # Use exuberant ctags to get the function names.
    ctags_tables: list[str] = []
    for file in GEMM_files:
        assert os.path.isfile(file), f"{file} does not exist"
        ctags_tables.append(os.popen("ctags -f- " + file).read())

    return get_functions_from_ctags_table("\n".join(ctags_tables))


GEMM_kernels: "set[str]" = get_GEMM_kernel_names()


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
    print(get_GEMM_kernel_names())
