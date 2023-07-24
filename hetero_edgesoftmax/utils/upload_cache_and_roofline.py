from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import ask_subdirectory
from .load_nsight_report import (
    extract_ncu_values_from_details,
    extract_ncu_values_from_raws,
    load_ncu_report,
    extract_csv_from_nsys_cli_output,
)
from typing import Union
from .upload_benchmark_results import NameCanonicalizer, update_gspread
import os


def extract_info_from_ncu(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.ncu-rep
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".ncu-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_info_from_nsys(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.nsys-rep
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".nsys-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_memory_from_ncu_folder(path) -> "list[list[Union[float, str, int]]]":
    results = []
    for filename in os.listdir(path):
        if filename.endswith(".ncu-rep"):
            name_and_info: list[str] = extract_info_from_ncu(filename)
            func_and_mem = extract_ncu_values_from_details(
                load_ncu_report(os.path.join(path, filename), "details")
            )
            results.append([name_and_info + f_ for f_ in func_and_mem])
    return results


def extract_roofline_from_ncu_folder(path) -> "list[list[Union[float, str, int]]]":
    results = []
    for filename in os.listdir(path):
        if filename.endswith(".ncu-rep"):
            name_and_info: list[str] = extract_info_from_ncu(filename)
            func_and_mem = extract_ncu_values_from_raws(
                load_ncu_report(os.path.join(path, filename), "raw")
            )
            results.append([name_and_info + f_ for f_ in func_and_mem])
    return results


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    dirname = ask_subdirectory("misc/artifacts", "ncu_breakdown_")

    pass
