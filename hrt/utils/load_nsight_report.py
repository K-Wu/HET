from .nsight_utils import (
    load_nsys_report as general_load_nsys_report,
    consolidate_ncu_details as general_consolidate_ncu_details,
)
from .classify_het_kernels import classify_het_kernel


def load_nsys_report(filename: str, report_name: str):
    return general_load_nsys_report(filename, report_name, classify_het_kernel)


def consolidate_ncu_details(
    metric_per_row: "list[list[str]]",
) -> "list[list[str]]":
    return general_consolidate_ncu_details(metric_per_row, classify_het_kernel)
