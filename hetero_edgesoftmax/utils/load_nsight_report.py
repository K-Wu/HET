import os
import re
from .detect_pwd import is_pwd_het_dev_root, run_once
from functools import lru_cache
from typing import Tuple
from .classify_het_kernels import classify_het_kernel


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
    # It seems arithmetic intensity is AchievedWorkPerSecond/BytesPerSecond
    # It is safe to duplicate entries because raw_metrics is a set
    raw_metrics: "set[str]" = {
        # Achieved work
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",  # value per cycle (1/3)
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",  # value per cycle (2/3)
        "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",  # Predicated-On FFMA Operations Per Cycle value per cycle (3/3)
        "smsp__cycles_elapsed.avg.per_second",  # "SM Frequency" cycle per second
        # L2 achieved traffic
        "l1tex__m_xbar2l1tex_read_bytes.sum.per_second",  # L2 Cache bandwidth achieved value
        # L1 achieved traffic
        "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second",  # L1 Cache Bandwidth (Global/Local) achieved traffic
        # DRAM achieved traffic
        "dram__bytes.sum.per_second",  # DRAM Bandwidth achieved value
        # Compute roofline
        "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",  # Theoretical Predicated-On FFMA Operations value per cycle
        "sm__cycles_elapsed.avg.per_second",  # "SM Frequency" cycle per second
        # DRAM roofline
        "dram__bytes.sum.peak_sustained",  # "Theoretical DRAM Bytes Accessible"
        "dram__cycles_elapsed.avg.per_second",  # DRAM frequency cycle per second
        # L1 roofline
        "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.peak_sustained",  # Theoretical L1/TEX Cache Bytes Accessible
        "l1tex__cycles_elapsed.avg.per_second",  # L1 cache frequency cycle per second
        # L2 roofline
        "l1tex__m_xbar2l1tex_read_bytes.sum.peak_sustained",  # "Theoretical L2 Cache Bytes Accessible" value per cycle
        "lts__cycles_elapsed.avg.per_second",  # "L2 cache frequency" cycle per second
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
    for idx in NCU_DETAILS_COLUMN_IDX.values():
        print(f"header[{idx}] = {header[idx]}, units[{idx}] = {units[idx]}")
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


def reorder_columns_in_raw_csv(
    kernel_instances_per_row: "list[list[str]]",
    metric_front: "list[Tuple[str,str]]",
    metric_end: "list[Tuple[str,str]]",
) -> "list[list[str]]":
    """
    Reorder the columns in raw csv so that the first few columns are those specified in front, and the last few columns are those specified in end
    """
    header: list[str] = kernel_instances_per_row[0]
    units: list[str] = kernel_instances_per_row[1]
    header_and_units: list[Tuple[str, str]] = [
        (header[i], units[i]) for i in range(len(header))
    ]
    kernel_identifier_columns: list[Tuple[str, str]] = [
        ("ID", ""),
        ("Pretty Name", ""),
        ("Kernel Name", ""),
    ]
    new_header_and_units: list[Tuple[str, str]] = (
        kernel_identifier_columns
        + metric_front
        + list(
            set(header_and_units)
            .difference(set(metric_front))
            .difference(set(metric_end))
            .difference(set(kernel_identifier_columns))
        )
        + metric_end
    )
    column_idx_to_original_idx: list[int] = [
        header_and_units.index(ele) for ele in new_header_and_units
    ]
    results: list[list[str]] = [
        [ele[0] for ele in new_header_and_units],
        [ele[1] for ele in new_header_and_units],
    ]
    for row in kernel_instances_per_row[2:]:
        results.append([row[idx] for idx in column_idx_to_original_idx])
    return results


def derive_rooflines(
    kernel_instances_metrics: dict[Tuple[str, str, str], dict[Tuple[str, str], str]],
    metrics_and_units: set[Tuple[str, str]],
) -> None:
    """compute rooflines and achieved values and add them to kernel_instances_metrics, and headers to metrics_and_units"""
    # TODO
    pass


def derive_kernel_categories(
    kernel_instances_metrics: dict[Tuple[str, str, str], dict[Tuple[str, str], str]],
    metrics_and_units: set[Tuple[str, str]],
) -> None:
    metrics_and_units.add(("Kernel Category", ""))
    for kernel_identifier in kernel_instances_metrics:
        kernel_instances_metrics[kernel_identifier][
            ("Kernel Category", "")
        ] = classify_het_kernel(
            kernel_identifier[2]
        )  # kernel_identifier[2] is kernel name


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
    kernel_instances_metrics: dict[
        Tuple[str, str, str], dict[Tuple[str, str], str]
    ] = {}

    metrics_and_units: set[Tuple[str, str]] = set()
    for row in metric_per_row[1:]:
        kernel_identifier: tuple[str, str, str] = (
            row[name_columns_idx["ID"]],
            row[name_columns_idx["Pretty Name"]],
            row[name_columns_idx["Kernel Name"]],
        )
        if kernel_identifier not in kernel_instances_metrics:
            kernel_instances_metrics[kernel_identifier] = dict()
        assert (
            row[metric_columns_idx["Metric Name"]],
            row[metric_columns_idx["Metric Unit"]],
        ) not in kernel_instances_metrics[kernel_identifier], f"Duplicate metric: {row}"

        kernel_instances_metrics[kernel_identifier][
            (
                row[metric_columns_idx["Metric Name"]],
                row[metric_columns_idx["Metric Unit"]],
            )
        ] = row[metric_columns_idx["Metric Value"]]

        metrics_and_units.add(
            (
                row[metric_columns_idx["Metric Name"]],
                row[metric_columns_idx["Metric Unit"]],
            )
        )

    derive_rooflines(kernel_instances_metrics, metrics_and_units)
    derive_kernel_categories(kernel_instances_metrics, metrics_and_units)
    # TODO: add support to metric orders in output

    results: list[list[str]] = [
        name_columns + [ele[0] for ele in metrics_and_units],
        [""] * len(name_columns) + [ele[1] for ele in metrics_and_units],
    ]
    for kernel_identifier in kernel_instances_metrics:
        row = list(kernel_identifier)
        for metric, unit in metrics_and_units:
            if (metric, unit) not in kernel_instances_metrics[kernel_identifier]:
                row.append("")
            else:
                row.append(kernel_instances_metrics[kernel_identifier][(metric, unit)])
        results.append(row)
    return results


def combine_ncu_raw_csvs(
    kernel_instances_metrics_list: "list[list[list[str]]]",
) -> list[list[str]]:
    """
    Combine multiple raw csvs from ncu into one
    """
    assert len(kernel_instances_metrics_list) > 0
    kernel_instances_metrics: dict[
        Tuple[str, str, str], dict[Tuple[str, str], str]
    ] = {}
    metrics_and_units: set[Tuple[str, str]] = set()
    for kernel_instances_metrics_ in kernel_instances_metrics_list:
        header = kernel_instances_metrics_[0]
        units = kernel_instances_metrics_[1]
        assert header[0] == "ID", f"header[0] = {header[0]} != ID"
        assert header[1] == "Pretty Name", f"header[1] = {header[1]} != Pretty Name"
        assert header[2] == "Kernel Name", f"header[2] = {header[2]} != Kernel Name"
        for row in kernel_instances_metrics_[2:]:
            kernel_identifier: tuple[str, str, str] = (
                row[0],
                row[1],
                row[2],
            )
            if kernel_identifier not in kernel_instances_metrics:
                kernel_instances_metrics[kernel_identifier] = dict()
            for metric_idx in range(3, len(row)):
                curr_metric = header[metric_idx]
                curr_unit = units[metric_idx]
                curr_value = row[metric_idx]
                if (curr_metric, curr_unit) not in kernel_instances_metrics[
                    kernel_identifier
                ]:
                    kernel_instances_metrics[kernel_identifier][
                        (curr_metric, curr_unit)
                    ] = curr_value
                    metrics_and_units.add((curr_metric, curr_unit))
                else:
                    print(
                        "Warning: duplicate metric",
                        curr_metric,
                        curr_unit,
                        curr_value,
                        kernel_identifier,
                        kernel_instances_metrics[kernel_identifier][
                            (curr_metric, curr_unit)
                        ],
                    )

    result_header = ["ID", "Pretty Name", "Kernel Name"] + [
        ele[0] for ele in metrics_and_units
    ]
    result_units = [""] * 3 + [ele[1] for ele in metrics_and_units]
    results: list[list[str]] = [result_header, result_units]
    for kernel_identifier in kernel_instances_metrics:
        row = list(kernel_identifier)
        for metric, unit in metrics_and_units:
            if (metric, unit) not in kernel_instances_metrics[kernel_identifier]:
                row.append("")
            else:
                row.append(kernel_instances_metrics[kernel_identifier][(metric, unit)])
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

    print(
        load_nsys_report(
            "utils/test/graphiler_hgt_fb15k.nsys-rep",
            "cuda_gpu_trace",
        )
    )

    print(
        load_nsys_report(
            "utils/test/graphiler_hgt_fb15k.nsys-rep",
            "cuda_gpu_trace,nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum",
        )
    )

    print(
        consolidate_ncu_details(
            extract_ncu_values_from_details(
                load_ncu_report(
                    "utils/test/HGT.aifb...64.64.1.ncu-rep",
                    "details",
                )
            )
        )
    )

    print(
        reorder_columns_in_raw_csv(
            extract_ncu_values_from_raws(
                load_ncu_report(
                    "utils/test/HGT.aifb...64.64.1.ncu-rep",
                    "raw",
                )
            ),
            metric_front=[],
            metric_end=[],
        )
    )

    extract_ncu_values_from_raws(
        load_ncu_report(
            "utils/test/HGT.aifb...64.64.1.ncu-rep",
            "raw",
        )
    )
