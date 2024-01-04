from .nsight_utils import (
    extract_csv_from_nsys_file as generic_extract_csv_from_nsys_file,
    consolidate_ncu_details as generic_consolidate_ncu_details,
)
from .classify_het_kernels import classify_het_kernel


def extract_csv_from_nsys_file(filename: str, report_name: str):
    return generic_extract_csv_from_nsys_file(
        filename, report_name, classify_het_kernel
    )


def consolidate_ncu_details(
    metric_per_row: "list[list[str]]",
) -> "list[list[str]]":
    return generic_consolidate_ncu_details(metric_per_row, classify_het_kernel)
