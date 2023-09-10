from typing import Tuple
from .load_nsight_report import load_nsys_report
from .nsight_utils import (
    prettify_name_from_func_signature,
)
from .classify_het_kernels import classify_het_kernel

# TODO: we assume n_warmups == 5, and n_epochs == 10, but we may in future get these parameters from arguments from the trace file
"""
After exporting to txt/json via nsys export -t json, the first type 27 event will involve argument information, e.g.,
Type: 27
CommEvent_ {
  Timestamp: -290185240
  GlobalPid: 341157691260928
  NumOfCpus: 0
  Command: "python"
  WorkDir: "/home/kunww/HET/hrt"
  Args: "-m"
  Args: "python.HGT.train"
  Args: "-d"
  Args: "am"
  Args: "--num_layers"
  Args: "1"
  Args: "--full_graph_training"
  Args: "--num_classes"
  Args: "64"
  Args: "--n_infeat"
  Args: "64"
  Args: "--num_heads"
  Args: "1"
  NsTime: true
  EnvironId: 17
}
"""


def get_avg_kern_sum(
    filepath: str, n_warmups=5, n_epochs=10
) -> "list[list[str]]":
    kern_traces: list[list[str]] = load_nsys_report(filepath, "cuda_gpu_trace")
    header = kern_traces[0]
    kernel_name_idx = header.index("Name")
    start_timestamp_idx = 0
    duration_idx = 1
    corr_id_idx = 2
    # Make sure the unit is nanoseconds
    assert header[start_timestamp_idx] == "Start (ns)"
    assert header[duration_idx] == "Duration (ns)"
    assert header[corr_id_idx] == "CorrId"
    kernel_instances: list[Tuple[str, int, str, int, int, str]] = []
    for line in kern_traces[1:]:
        # Each line stores a kernel instance, i.e., launch.
        kernel_name = line[kernel_name_idx]
        pretty_name = prettify_name_from_func_signature(kernel_name)
        duration = int(line[duration_idx])
        start_timestamp = int(line[start_timestamp_idx])
        corr_id = int(line[corr_id_idx])
        if classify_het_kernel(pretty_name) == "Non-HET Others":
            continue
        kernel_instances.append(
            (
                kernel_name,
                corr_id,
                pretty_name,
                duration,
                start_timestamp,
                classify_het_kernel(pretty_name),
            )
        )

    # TODO: devise algorithm to figure out the beginning id of trainning, i.e., after warm up ends
    # training_beg_idx = len(kernel_instances)
    results_csv: list[list[str]] = [
        [
            "Kernel name",
            "CorrId",
            "Pretty name",
            "Duration (ns)",
            "Start timestamp (ns)",
            "Kernel type",
        ]
    ]
    results_csv += [list(map(str, row)) for row in kernel_instances]
    return results_csv
