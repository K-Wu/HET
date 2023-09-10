if __name__ == "__main__":
    from .detect_pwd import is_pwd_het_dev_root, RESULTS_RELATIVE_DIR
    from .nsight_utils import upload_nsys_report
    from .nsight_utils import (
        ask_subdirectory_or_file,
    )
    from .classify_het_kernels import classify_het_kernel
    from .upload_benchmark_results import SPREADSHEET_URL

    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    path_name = ask_subdirectory_or_file(
        "misc/artifacts", "nsys_trace_", RESULTS_RELATIVE_DIR
    )
    upload_nsys_report(
        path_name,
        "cuda_kern_exec_trace",
        SPREADSHEET_URL,
        classify_het_kernel,
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )
