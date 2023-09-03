from .detect_pwd import get_het_root_path
from .nsight_utils.classify_het_kernels import (
    get_GEMM_kernel_names,
    is_ctags_installed,
    get_fw_bw_host_func_names,
    classify_fw_bw_kernel,
)

GEMM_kernels: "set[str]" = get_GEMM_kernel_names(get_het_root_path())


def test_classify_fw_bw_kernel():
    kernel_names = set()
    with open(
        f"{get_het_root_path()}/hetero_edgesoftmax/utils/test/kernel_names_trace.test_log"
    ) as f:
        for line in f:
            kernel_names.add(line.strip())

    for func_pretty_name in kernel_names:
        if classify_fw_bw_kernel(func_pretty_name) == "FwProp":
            continue
        elif classify_fw_bw_kernel(func_pretty_name) == "BckProp":
            continue
        else:
            print(f"{func_pretty_name} is neither a forward nor a backward kernel")
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
