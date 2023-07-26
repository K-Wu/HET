from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import ask_subdirectory_or_file
from .load_nsight_report import (
    extract_ncu_values_from_details,
    extract_ncu_values_from_raws,
    load_ncu_report,
    extract_csv_from_nsys_cli_output,
    calculate_roofline_for_ncu_raw_csvs,
    combine_ncu_raw_csvs,
    consolidate_ncu_details,
)
from .upload_benchmark_results import (
    NameCanonicalizer,
    update_gspread,
    SPREADSHEET_URL,
    create_worksheet,
)
import os
import socket


def extract_info_from_ncu(file_path: str) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.ncu-rep
    file_path = os.path.basename(file_path)
    return NameCanonicalizer.to_list(
        file_path[: file_path.rfind(".ncu-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_info_from_nsys(file_path: str) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.nsys-rep
    file_path = os.path.basename(file_path)
    return NameCanonicalizer.to_list(
        file_path[: file_path.rfind(".nsys-rep")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_from_ncu_file(
    file_path: str, extract_mem_flag: bool, extract_roofline_flag: bool
) -> "list[list[str]]":
    assert file_path.endswith(".ncu-rep"), "filename must end with .ncu-rep"
    name_and_info: list[str] = extract_info_from_ncu(file_path)
    func_and_metric_csvs: list[list[list[str]]] = []
    if extract_mem_flag:
        func_and_metric_csvs.append(
            consolidate_ncu_details(
                extract_ncu_values_from_details(load_ncu_report(file_path, "details"))
            )
        )
    if extract_roofline_flag:
        func_and_metric_csvs.append(
            calculate_roofline_for_ncu_raw_csvs(
                extract_ncu_values_from_raws(load_ncu_report(file_path, "raw"))
            )
        )
    if len(func_and_metric_csvs) == 1:
        results = [name_and_info + f_ for f_ in func_and_metric_csvs[0]]
    else:
        # number of frozen columns is 3, i.e., (id, pretty name, kernel name)
        # names and infos will be added after the combination
        func_and_metric = combine_ncu_raw_csvs(3, func_and_metric_csvs)
        results = [name_and_info + f_ for f_ in func_and_metric]
    return results


def extract_from_ncu_folder(
    path: str, extract_mem_flag: bool, extract_roofline_flag: bool
) -> "list[list[str]]":
    raw_csvs: list[list[list[str]]] = []
    len_name_and_info: int = -1
    for filename in os.listdir(path):
        print("extract_from_ncu_folder Processing", filename)
        if filename.endswith(".ncu-rep"):
            raw_csvs.append(
                extract_from_ncu_file(
                    os.path.join(path, filename),
                    extract_mem_flag,
                    extract_roofline_flag,
                )
            )
        if (
            len(extract_info_from_ncu(filename)) != len_name_and_info
            and len_name_and_info != -1
        ):
            raise ValueError("Number of frozen columns not consistent")
        len_name_and_info = len(extract_info_from_ncu(filename))
    # number of frozen columns equals to the number of columns in name_and_info and (id, pretty name, kernel name)

    return combine_ncu_raw_csvs(len_name_and_info + 3, raw_csvs)


def extract_memory_from_ncu_folder(path: str) -> "list[list[str]]":
    return extract_from_ncu_folder(
        path, extract_mem_flag=True, extract_roofline_flag=False
    )


def extract_roofline_from_ncu_folder(path: str) -> "list[list[str]]":
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
    path_name = ask_subdirectory_or_file("misc/artifacts", "ncu_breakdown_")
    if os.path.isdir(path_name):
        csv_rows = extract_from_ncu_folder(path_name, True, True)
    else:
        csv_rows = extract_from_ncu_file(path_name, True, True)

    print(csv_rows)

    worksheet_title = f"[{socket.gethostname()}]{path_name.split('/')[-1]}"[:100]
    try:
        update_gspread(
            csv_rows,
            create_worksheet(SPREADSHEET_URL, worksheet_title)
            # open_worksheet(SPREADSHEET_URL, "0") # GID0 reserved for testing
        )
    except Exception as e:
        print("Failed to upload ncu results:", e)
