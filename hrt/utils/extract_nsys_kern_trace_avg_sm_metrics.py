"""Print the average SM metric of the last code range of the specified nsys profiling result"""
if __name__ == "__main__":
    from .detect_pwd import is_pwd_het_dev_root, RESULTS_DIR
    from .nsight_utils import ask_subdirectory_or_file, upload_nsys_reports
    from .upload_benchmark_results import SPREADSHEET_URL
    from .classify_het_kernels import classify_het_kernel
    from .nsight_utils import (
        get_last_nvtx_range,
        calc_avg_sm_metrics,
        get_kern_trace_overhead,
    )
    import os

    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    dirname = ask_subdirectory_or_file(
        "misc/artifacts",
        "motivator_graphiler_breakdown_",
        RESULTS_DIR,
    )
    for subdir_or_file in os.listdir(dirname):
        if os.path.isfile(subdir_or_file):
            file_path = os.path.join(dirname, subdir_or_file)
            start, duration = get_last_nvtx_range(file_path)

    # upload_nsys_reports(
    #     path_name,
    #     "cuda_kern_exec_sum",
    #     SPREADSHEET_URL,
    #     classify_het_kernel,
    #     "model.dataset.bg.breakdown",
    # )
