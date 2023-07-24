from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import ask_subdirectory
from .load_nsight_report import (
    extract_ncu_values_from,
    extract_csv_from_nsys_cli_output,
)
from .upload_benchmark_results import NameCanonicalizer


def extract_info_from_ncu(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.resul
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".ncu-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_info_from_nsys(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.resul
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".nsys-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    dirname = ask_subdirectory("misc/artifacts", "ncu_breakdown_")
    pass
