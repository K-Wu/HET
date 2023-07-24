import os
import re
from .detect_pwd import is_pwd_het_dev_root, run_once
from functools import lru_cache


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
    nsys_cli_output: str = os.popen(
        f"nsys stats -f csv -r {report_name} {filename}"
    ).read()
    return extract_csv_from_nsys_cli_output(nsys_cli_output)


NCU_DETAILS_COLUMN_IDX = {
    "Kernel Name": 4,
    "Section Name": 11,
    "Metric Name": 12,
    "Metric Unit": 13,
    "Metric Value": 14,
    "Rule Name": 15,
    "Rule Type": 16,
    "Rule Description": 17,
}


def extract_ncu_values_from(
    nsys_details_csv: "list[list[str]]",
    metric_names: "set[str]" = {
        "L2 Cache Throughput",
        "L1/TEX Cache Throughput",
        "DRAM Throughput",
        "Memory Throughput",
    },
):
    header = nsys_details_csv[0]

    results = []
    for key in NCU_DETAILS_COLUMN_IDX:
        assert (
            header[NCU_DETAILS_COLUMN_IDX[key]] == key
        ), f"header[{NCU_DETAILS_COLUMN_IDX[key]}] = {header[NCU_DETAILS_COLUMN_IDX[key]]} != {key}"
    for row in nsys_details_csv[1:]:
        if row[NCU_DETAILS_COLUMN_IDX["Metric Name"]] in metric_names:
            results.append(
                [
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
    lines = csv_string.split("\n")
    for line in lines:
        line = line.strip()
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
            "../third_party/OthersArtifacts/graphiler/graphiler_hgt_fb15k.nsys-rep",
            "cuda_gpu_trace",
        )
    )
    print(
        load_nsys_report(
            "../third_party/OthersArtifacts/graphiler/graphiler_hgt_fb15k.nsys-rep",
            "cuda_gpu_trace,nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum",
        )
    )

    print(
        extract_ncu_values_from(
            load_ncu_report(
                "../third_party/OthersArtifacts/graphiler/hgt_biokg.ncu-rep", "details"
            )
        )
    )
