from .nsight_utils import (
    update_gspread,
    create_worksheet,
    get_cell_range_from_A1,
    count_cols,
    count_rows,
    get_pretty_hostname,
    find_latest_subdirectory_or_file,
    NameCanonicalizer,
)
import os
from typing import Union

import traceback

SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1qMklewOvYRVRHTYlMErvyd67afJvaVNwd79sMrKw__4/"
WORKSHEET_GIDS = [
    893574800,
    721906000,
    1311240734,
    478745927,
    1807654113,
    802529064,
    1406431095,
    1558424981,
    75558257,
]


def extract_info_from(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.result.log
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".result.log")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


def extract_result_from_graphiler_log(
    file_path: str,
) -> "list[list[Union[float,str,int]]]":
    result: list[list[Union[float, str, int]]] = []

    def get_infer_or_training(experiment_unit: str) -> str:
        # experiment_unit is either ms/infer or ms/training
        if experiment_unit == "ms/training":
            return "training"
        elif experiment_unit == "ms/infer":
            return "inference"
        raise ValueError(f"Unknown experiment_unit {experiment_unit}")

    with open(file_path, "r") as f:
        curr_dataset_name = ""
        lines = [l.strip() for l in f.readlines()]
        for idx_line, line in enumerate(lines):
            if line.find("benchmarking on") == 0:
                curr_dataset_name = line.split(" ")[-1].strip()
            elif line.find("elapsed time:") != -1:
                # Detect silent error by checking if memory use == 0.0 MB
                assert lines[idx_line + 1].find("memory usage:") != -1
                memory_use = float(
                    lines[idx_line + 1].split(" ")[-2].strip()
                )  # remove "MB"
                if memory_use == 0.0:
                    # Skipping
                    continue
                experiment_name = line.split(" ")[0].strip()
                experiment_unit = line.split(" ")[-1].strip()
                experiment_value = float(line.split(" ")[-2].strip())
                result.append(
                    [
                        curr_dataset_name,
                        experiment_name,
                        get_infer_or_training(experiment_unit),
                        experiment_value,
                    ]
                )
    return result


def extract_graphiler_and_its_baselines_results_from_folder(
    results_dir: str,
) -> "list[list[Union[float, str, int]]]":
    result: list[list[Union[float, str, int]]] = []
    for model in ["HGT", "RGAT", "RGCN"]:
        curr_result = extract_result_from_graphiler_log(
            os.path.join(results_dir, model + ".log")
        )
        curr_result = [[model, "graphiler"] + row for row in curr_result]
        result += curr_result
        curr_result = extract_result_from_graphiler_log(
            os.path.join(results_dir, model + "_baseline_standalone.log")
        )
        curr_result = [[model, "baselines"] + row for row in curr_result]
        result += curr_result
    return result


def extract_het_results_from_folder(
    path,
) -> "list[list[Union[float, str, int]]]":
    """
    Find and extract the following patterns.
    "Mean forward time: {:4f} ms"
    "Mean backward time: {:4f} ms"
    "Mean training time: {:4f} ms"
    OUTPUT_DIR="misc/artifacts/benchmark_all_`date +%Y%m%d%H%M`"
    """
    all_names_and_info = []
    for filename in os.listdir(path):
        if filename.endswith(".result.log"):
            name_info = extract_info_from(filename)
            avg_forward_time = "NotFound"
            avg_backward_time = "NotFound"
            avg_training_time = "NotFound"
            status = []
            with open(os.path.join(path, filename), "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Mean forward time"):
                        avg_forward_time = float(
                            line.split(":")[1].strip().split()[0]
                        )
                    elif line.startswith("Mean backward time"):
                        avg_backward_time = float(
                            line.split(":")[1].strip().split()[0]
                        )
                    elif line.startswith("Mean training time"):
                        avg_training_time = float(
                            line.split(":")[1].strip().split()[0]
                        )
                    if line.lower().find("error") != -1:
                        status.append(line.strip())
            if len(status) == 0:
                if "NotFound" in [
                    avg_forward_time,
                    avg_backward_time,
                    avg_training_time,
                ]:
                    status = ["Silent Error (Likely OOM or SEGV)"]
                else:
                    status = ["OK"]
            status_str = "; ".join(status)
            name_info += [
                avg_forward_time,
                avg_backward_time,
                avg_training_time,
                status_str,
            ]
            all_names_and_info.append(name_info)
    return all_names_and_info


def upload_folder(
    root: str,
    prefix: str,
    is_graphiler_flag: bool,
    test_repeat_x_y: bool = False,
):
    dir_to_upload = find_latest_subdirectory_or_file(root, prefix)
    print("Uploading results from", dir_to_upload)
    if is_graphiler_flag:
        names_and_info = (
            extract_graphiler_and_its_baselines_results_from_folder(
                dir_to_upload
            )
        )
    else:
        names_and_info = extract_het_results_from_folder(dir_to_upload)
    print(names_and_info)
    worksheet_title = (
        f"[{get_pretty_hostname()}]{dir_to_upload.split('/')[-1]}"
    )
    try:
        worksheet = create_worksheet(SPREADSHEET_URL, worksheet_title)
        if not test_repeat_x_y:
            update_gspread(
                names_and_info,
                worksheet
                # open_worksheet(SPREADSHEET_URL, "0") # GID0 reserved for testing
            )
        else:  # Repeat once in each dimension to test the indexing scheme
            update_gspread(
                names_and_info,
                worksheet,
                cell_range=get_cell_range_from_A1(
                    count_rows(names_and_info),
                    count_cols(names_and_info),
                    0,
                    0,
                ),
            )
            update_gspread(
                names_and_info,
                worksheet,
                cell_range=get_cell_range_from_A1(
                    count_rows(names_and_info),
                    count_cols(names_and_info),
                    count_rows(names_and_info),
                    0,
                ),
            )
            update_gspread(
                names_and_info,
                worksheet,
                cell_range=get_cell_range_from_A1(
                    count_rows(names_and_info),
                    count_cols(names_and_info),
                    0,
                    count_cols(names_and_info),
                ),
            )
    except Exception as e:
        print("Failed to upload results:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    from .detect_pwd import is_pwd_het_dev_root

    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    upload_folder("misc/artifacts", "graphiler_", True, False)
    upload_folder("misc/artifacts", "benchmark_all_", False, False)
