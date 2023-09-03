from .nsight_utils.load_nsight_report import (
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


if __name__ == "__main__":
    from .detect_pwd import is_pwd_het_dev_root
    from .nsight_utils.load_nsight_report import (
        extract_ncu_values_from_details,
        load_ncu_report,
        reorder_columns_in_raw_csv,
        extract_ncu_values_from_raws,
        calculate_roofline_for_ncu_raw_csvs,
    )

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

    print(
        calculate_roofline_for_ncu_raw_csvs(
            extract_ncu_values_from_raws(
                load_ncu_report(
                    "utils/test/HGT.aifb...64.64.1.ncu-rep",
                    "raw",
                )
            )
        )[0:2]
    )
