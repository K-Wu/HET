import os
import re
from .detect_pwd import is_pwd_het_dev_root, run_once
from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=None)
@run_once
def nsys_exists() -> bool:
    """Check if nsys is installed."""
    return os.system("nsys --version >/dev/null 2>/dev/null") == 0


@lru_cache(maxsize=None)
@run_once
def ncu_exists() -> bool:
    """Check if ncu is installed."""
    return os.system("ncu --version >/dev/null 2>/dev/null") == 0


def extract_csv_from_nsys_cli_output(nsys_cli_output: str) -> "list[list[str]]":
    """Extract csv from nsys cli output."""
    lines = nsys_cli_output.split("\n")
    result = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        elif re.match(
            r"Processing \[([\.\w\/\-])+\] with \[([\.\w\/\-])+\]\.\.\.", line
        ):
            continue
        else:
            # print(line)
            result.append(line.split(","))
    return result


def load_nsys_report(filename: str, report_name: str) -> "list[list[str]]":
    """Load a report from a nsys report file."""
    assert nsys_exists(), "nsys is not installed"
    assert os.path.exists(filename), f"{filename} does not exist"
    nsys_cli_output: str = os.popen(
        f"nsys stats -f csv -r {report_name} {filename}"
    ).read()
    return extract_csv_from_nsys_cli_output(nsys_cli_output)


NCU_DETAILS_COLUMN_IDX = {
    "ID": 0,
    "Kernel Name": 4,
    "Section Name": 11,
    "Metric Name": 12,
    "Metric Unit": 13,
    "Metric Value": 14,
    "Rule Name": 15,
    "Rule Type": 16,
    "Rule Description": 17,
}


def construct_raw_column_idx(
    header: "list[str]", columns: "set[str]"
) -> "dict[str, int]":
    result = {}
    for column in columns:
        result[column] = header.index(column)
    return result


def extract_ncu_values_from_raws(
    nsys_details_csv: "list[list[str]]",
    raw_metrics: "set[str]" = {
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
        "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
        "smsp__cycles_elapsed.avg.per_second",
        "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second",
        "l1tex__m_xbar2l1tex_read_bytes.sum.per_second",
        "dram__bytes.sum.per_second",
        "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
        "smsp__cycles_elapsed.avg.per_second",
    },
) -> "list[list[str]]":
    header: list[str] = nsys_details_csv[0]
    units: list[str] = nsys_details_csv[1]
    assert header[0] == "ID", f"header[0] = {header[0]} != ID"
    assert header[4] == "Kernel Name", f"header[4] = {header[4]} != Kernel Name"
    NCU_DETAILS_COLUMN_IDX: dict[str, int] = construct_raw_column_idx(
        header,
        raw_metrics,
    )
    results: list[list[str]] = [
        ["ID", "Pretty Name", "Kernel Name"] + [key for key in NCU_DETAILS_COLUMN_IDX],
        ["", "", ""]
        + [units[NCU_DETAILS_COLUMN_IDX[key]] for key in NCU_DETAILS_COLUMN_IDX],
    ]
    for row in nsys_details_csv[2:]:  # Skip header and units
        results.append(
            [
                row[0],  # ID
                extract_func_name_from_signature(row[4]),  # Kernel Name
                row[4],  # Kernel Name
            ]
            + [row[NCU_DETAILS_COLUMN_IDX[key]] for key in NCU_DETAILS_COLUMN_IDX]
        )
    return results


def duplicate_metrics_exists(metrics_and_units: "set[Tuple[str, str]]") -> bool:
    """
    Assert there are no two metrics with the same name (i.e., with different units)
    """
    metrics: set[str] = set()
    for metric, unit in metrics_and_units:
        if metric in metrics:
            return True  # f"Duplicate metric {metric} with different units"
        metrics.add(metric)
    return False


def consolidate_ncu_details(metric_per_row: "list[list[str]]") -> "list[list[str]]":
    """
    The original output from extract_ncu_values_from_details shows one metric in each row,
    this function consolidate it so that each row show all metrics of a kernel instance,
    similar to the ncu raw csv output
    """
    header: list[str] = metric_per_row[0]
    name_columns: list[str] = ["ID", "Pretty Name", "Kernel Name"]
    name_columns_idx: dict[str, int] = {key: header.index(key) for key in name_columns}
    metric_columns: list[str] = ["Metric Name", "Metric Unit", "Metric Value"]
    metric_columns_idx: dict[str, int] = {
        key: header.index(key) for key in metric_columns
    }
    results_dict: dict[Tuple[str, str, str], dict[str, Tuple[str, str]]] = {}

    metrics_and_units: set[Tuple[str, str]] = {}
    for row in metric_per_row[1:]:
        key = (
            row[name_columns_idx["ID"]],
            row[name_columns_idx["Pretty Name"]],
            row[name_columns_idx["Kernel Name"]],
        )
        if key not in results_dict:
            results_dict[key] = dict()
        assert (
            row[metric_columns_idx["Metric Name"]] not in results_dict[key]
        ), f"Duplicate metric name {row[metric_columns_idx['Metric Name']]} for {key}"
        results_dict[key][row[metric_columns_idx["Metric Name"]]] = (
            row[metric_columns_idx["Metric Unit"]],
            row[metric_columns_idx["Metric Value"]],
        )
        metrics_and_units.add(
            (
                row[metric_columns_idx["Metric Name"]],
                row[metric_columns_idx["Metric Unit"]],
            )
        )

    assert not duplicate_metrics_exists(
        metrics_and_units
    ), f"Duplicate metrics exist: {metrics_and_units}"
    results: list[list[str]] = [
        name_columns + [ele[0] for ele in metrics_and_units],
        [""] * len(name_columns) + [ele[1] for ele in metrics_and_units],
    ]
    for key in results_dict:
        row = list(key)
        for metric in metric_columns:
            # print("Warning: ", metric, "not in", key)
            row += results_dict[key][metric]
        results.append(row)
    return results


def extract_ncu_values_from_details(
    nsys_details_csv: "list[list[str]]",
    metric_names: "set[str]" = {
        "ID",
        "L2 Cache Throughput",
        "L1/TEX Cache Throughput",
        "DRAM Throughput",
        "Memory Throughput",
    },
) -> "list[list[str]]":
    header: list[str] = nsys_details_csv[0]

    results: list[list[str]] = [
        [
            "ID",
            "Pretty Name",
            "Kernel Name",
            "Metric Name",
            "Metric Unit",
            "Metric Value",
        ]
    ]
    for key in NCU_DETAILS_COLUMN_IDX:
        assert (
            header[NCU_DETAILS_COLUMN_IDX[key]] == key
        ), f"header[{NCU_DETAILS_COLUMN_IDX[key]}] = {header[NCU_DETAILS_COLUMN_IDX[key]]} != {key}"
    for row in nsys_details_csv[1:]:
        if row[NCU_DETAILS_COLUMN_IDX["Metric Name"]] in metric_names:
            results.append(
                [
                    row[NCU_DETAILS_COLUMN_IDX["ID"]],
                    extract_func_name_from_signature(
                        row[NCU_DETAILS_COLUMN_IDX["Kernel Name"]]
                    ),
                    row[NCU_DETAILS_COLUMN_IDX["Kernel Name"]],
                    row[NCU_DETAILS_COLUMN_IDX["Metric Name"]],
                    row[NCU_DETAILS_COLUMN_IDX["Metric Unit"]],
                    row[NCU_DETAILS_COLUMN_IDX["Metric Value"]],
                ]
            )
    return results


def load_csv_from_multiline_string(csv_string: str) -> "list[list[str]]":
    # ncu output is multiline csv where each cell value is wrapped by double quotes.
    # We need to remove the double quotes and split the string by comma.
    result = []
    lines: list[str] = csv_string.split("\n")
    for line in lines:
        line: str = line.strip()
        if len(line) == 0:
            continue
        elif line.startswith('"'):
            if line.endswith('"'):
                result.append(line[1:-1].split('","'))
            elif line.endswith('",'):
                result.append(line[1:-2].split('","'))
            continue

        print('Warning: line does not start with " or end with ",:', line)
    return result


def load_ncu_report(filename: str, page_name: str) -> "list[list[str]]":
    """Load a report from a ncu report file."""
    assert ncu_exists(), "ncu is not installed"
    assert os.path.exists(filename), f"{filename} does not exist"
    nsys_cli_output: str = os.popen(
        f"ncu --page {page_name} --csv  --import {filename}"
    ).read()
    return load_csv_from_multiline_string(nsys_cli_output)


def extract_func_name_from_signature(func_signature: str) -> str:
    # func_signature: HET_XXX<XX,XX,XX>(XXX, XXX, XXX)
    result = (
        func_signature.split("(")[0]
        .strip()
        .split("<")[0]
        .strip()
        .split(" ")[-1]
        .strip()
    )
    if result == "cutlass::Kernel":
        return (
            func_signature.split("(")[0].strip().split("<")[1].strip()[:-1]
        )  # remove the last >
    else:
        return result


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    try:
        print(
            load_nsys_report(
                "utils/test/graphiler_hgt_fb15k.nsys-rep",
                "cuda_gpu_trace",
            )
        )
    except Exception as e:
        print("Error occurred", e)

    try:
        print(
            load_nsys_report(
                "utils/test/graphiler_hgt_fb15k.nsys-rep",
                "cuda_gpu_trace,nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum",
            )
        )
    except Exception as e:
        print("Error occurred2", e)

    try:
        print(
            extract_ncu_values_from_details(
                load_ncu_report(
                    "../third_party/OthersArtifacts/graphiler/graphiler.bgs.HGT.64.64.ncu-rep",
                    "details",
                )
            )
        )
    except Exception as e:
        print("Error occurred3", e)
