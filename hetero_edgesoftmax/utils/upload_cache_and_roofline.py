from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import ask_subdirectory
from .load_nsight_report import (
    extract_ncu_values_from_details,
    extract_ncu_values_from_raws,
    load_ncu_report,
    extract_csv_from_nsys_cli_output,
    combine_ncu_raw_csvs,
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


def extract_from_ncu_folder(
    path: str, extract_mem_flag: bool, extract_roofline_flag: bool
) -> "list[list[Union[float, str, int]]]":
    results = []
    for filename in os.listdir(path):
        if filename.endswith(".ncu-rep"):
            name_and_info: list[str] = extract_info_from_ncu(filename)
            func_and_metric_list: list[list[list[str]]] = []
            if extract_mem_flag:
                func_and_metric_list.append(
                    extract_ncu_values_from_details(
                        load_ncu_report(os.path.join(path, filename), "details")
                    )
                )
            if extract_roofline_flag:
                func_and_metric_list.append(
                    extract_ncu_values_from_raws(
                        load_ncu_report(os.path.join(path, filename), "raw")
                    )
                )
            if len(func_and_metric_list) == 1:
                results.append([name_and_info + f_ for f_ in func_and_metric_list[0]])
            else:
                func_and_metric = combine_ncu_raw_csvs(func_and_metric_list)
                results.append([name_and_info + f_ for f_ in func_and_metric])

    return results


def extract_memory_from_ncu_folder(path: str) -> "list[list[Union[float, str, int]]]":
    return extract_from_ncu_folder(
        path, extract_mem_flag=True, extract_roofline_flag=False
    )


def extract_roofline_from_ncu_folder(path: str) -> "list[list[Union[float, str, int]]]":
    return extract_from_ncu_folder(
        path, extract_mem_flag=False, extract_roofline_flag=True
    )


def check_metric_units_all_identical_from_ncu_folder(path) -> bool:
    """
    check_metric_units_all_identical_from_ncu_folder("misc/artifacts/ncu_breakdown_202307180518") returns False after printing
    Metric derived__memory_l1_wavefronts_shared_excessive has different units: {'Kbyte', 'byte', 'Mbyte'}
    """
    metric_units: dict[str, set[str]] = dict()
    for filename in os.listdir(path):
        if filename.endswith(".ncu-rep"):
            raw_csv: list[list[str]] = load_ncu_report(
                os.path.join(path, filename), "raw"
            )

            for idx in range(len(raw_csv[0])):
                metric: str = raw_csv[0][idx]
                unit: str = raw_csv[1][idx]
                if metric not in metric_units:
                    metric_units[metric] = set()
                metric_units[metric].add(unit)

    for metric in metric_units:
        if len(metric_units[metric]) != 1:
            if len(metric_units[metric]) == 2 and "%" in metric_units[metric]:
                continue
            print(f"Metric {metric} has different units: {metric_units[metric]}")
            return False
    return True


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    dirname = ask_subdirectory("misc/artifacts", "ncu_breakdown_")
    print(check_metric_units_all_identical_from_ncu_folder(dirname))
    pass
