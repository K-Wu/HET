import os
import re
from .detect_pwd import is_pwd_het_dev_root
import os


def is_nsys_exist() -> bool:
    """Check if nsys is installed."""
    return os.system("nsys --version >/dev/null 2>/dev/null") == 0


def extract_csv_from_nsys_cli_output(nsys_cli_output: str) -> list[list[str]]:
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


def load_nsys_report(filename: str, report_name: str) -> list[list[str]]:
    """Load a report from a nsys report file."""
    assert is_nsys_exist(), "nsys is not installed"
    nsys_cli_output: str = os.popen(
        f"nsys stats -f csv -r {report_name} {filename}"
    ).read()
    return extract_csv_from_nsys_cli_output(nsys_cli_output)


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
