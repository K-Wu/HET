from .detect_pwd import is_pwd_het_dev_root
from .upload_motivator_graphiler_breakdown import upload_nsys_report
from .upload_benchmark_results import ask_subdirectory_or_file


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    path_name = ask_subdirectory_or_file("misc/artifacts", "nsys_trace_")
    upload_nsys_report(path_name, "cuda_kern_exec_trace")
