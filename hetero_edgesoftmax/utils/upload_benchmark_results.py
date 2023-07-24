# Some code is from https://github.com/COVID19Tracking/ltc-data-processing
# And https://github.com/nlioc4/FSBot/blob/f7f1a000ec7d02056c136fe68b7f0ca2271c80ae/modules/accounts_handler.py#L326
# To create a credential, or set up a new spreasheet, follow instruction at https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem
from typing import Union
from .detect_pwd import is_pwd_het_dev_root

import os
import socket

# TODO: find and extract the following pattern from the result folder
#
# "Mean forward time: {:4f} ms"
# "Mean backward time: {:4f} ms"
# "Mean training time: {:4f} ms"
# OUTPUT_DIR="misc/artifacts/benchmark_all_`date +%Y%m%d%H%M`"
#

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


class ConfigCanonicalizer:
    @classmethod
    def permute(cls, input_fmt: str, config: "list[str]") -> "list[str]":
        """
        sort the config list according to reverse-alphabetical order of input_fmt
        """
        input_fmt_: list[str] = input_fmt.split(".")
        assert len(input_fmt_) == len(config)
        return [
            c
            for _, c in sorted(
                zip(input_fmt_, config), key=lambda pair: pair[0], reverse=True
            )
        ]

    @classmethod
    def to_list(cls, config: "list[str]", input_fmt: str) -> "list[str]":
        if input_fmt is not None:
            config = cls.permute(input_fmt, config)
        return [c[2:] if c.startswith("--") else c for c in config]

    @classmethod
    def to_str(cls, config: "list[str]", input_fmt: str) -> str:
        return ".".join(cls.to_list(config, input_fmt))


class NameCanonicalizer:
    @classmethod
    def to_list(cls, name: str, input_fmt: str) -> "list[str]":
        input_fmt_: list[str] = input_fmt.split(".")
        name_: list[str] = name.split(".")
        assert len(input_fmt_) == len(name_)
        config_fmt = ".".join(
            [ele for ele in input_fmt_ if ele not in {"model", "dataset"}]
        )
        model = name_[input_fmt_.index("model")]
        dataset = name_[input_fmt_.index("dataset")]
        configs = [
            name_[idx]
            for idx in range(len(name_))
            if input_fmt_[idx] not in {"model", "dataset"}
        ]
        return [model, dataset] + ConfigCanonicalizer.to_list(configs, config_fmt)

    @classmethod
    def to_str(cls, name: str, input_fmt: str) -> str:
        return ".".join(cls.to_list(name, input_fmt))


def extract_info_from(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.result.log
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".result.log")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )
    # model, dataset = filename.split(".")[:2]
    # config: list[str] = filename.split(".")[2:-2]

    # return [model, dataset] + ConfigCanonicalizer.to_list(
    #     config, input_fmt="flag_mul.flag_compact.ax_in.ax_out.ax_head"
    # )


def find_latest_subdirectory(root, prefix) -> str:
    candidates = []
    for subdir in os.listdir(root):
        if subdir.startswith(prefix):
            candidates.append(subdir)
    return os.path.join(root, max(candidates))


def ask_subdirectory_or_file(root, prefix) -> str:
    """
    Show latest directory and request user input
    If user input is empty then choose the latest directory
    otherwise, choose the user input
    """
    candidate = find_latest_subdirectory(root, prefix)
    print(
        "With prefix ",
        prefix,
        ", the latest directory is ",
        os.path.basename(candidate),
    )
    user_input = input(
        "Press enter to use it, or please input the directory you want to upload: "
    )
    if len(user_input) == 0:
        result = candidate
    else:
        result = os.path.join(root, user_input)
    assert os.path.exists(result), f"{result} does not exist"
    return result


def extract_result_from_graphiler_log(
    file_path: str,
) -> "list[list[Union[float,str,int]]]":
    result: list[list[Union[float, str, int]]] = []
    with open(file_path, "r") as f:
        curr_dataset_name = ""
        for line in f:
            if line.find("benchmarking on") == 0:
                curr_dataset_name = line.split(" ")[-1].strip()
            elif line.find("elapsed time:") != -1:
                experiment_name = line.split(" ")[0].strip()
                experiment_unit = line.split(" ")[-1].strip()
                experiment_value = float(line.split(" ")[-2].strip())
                result.append(
                    [
                        curr_dataset_name,
                        experiment_name + "," + experiment_unit,
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


def extract_het_results_from_folder(path) -> "list[list[Union[float, str, int]]]":
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
                        avg_forward_time = float(line.split(":")[1].strip().split()[0])
                    elif line.startswith("Mean backward time"):
                        avg_backward_time = float(line.split(":")[1].strip().split()[0])
                    elif line.startswith("Mean training time"):
                        avg_training_time = float(line.split(":")[1].strip().split()[0])
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


def open_worksheet(target_sheet_url: str, target_gid: str):
    if target_gid != "0":
        raise NotImplementedError("To avoid data loss, only gid=0 is supported for now")
    gc = gspread.service_account()
    sh = gc.open_by_url(target_sheet_url)
    sheet_data = sh.fetch_sheet_metadata()

    try:
        item = finditem(
            lambda x: str(x["properties"]["sheetId"]) == target_gid,
            sheet_data["sheets"],
        )
        ws = Worksheet(sh, item["properties"])
    except (StopIteration, KeyError):
        raise WorksheetNotFound(target_gid)
    return ws


def create_worksheet(target_sheet_url: str, title: str) -> Worksheet:
    gc = gspread.service_account()
    sh = gc.open_by_url(target_sheet_url)
    return sh.add_worksheet(title=title, rows=100, cols=20)


def update_gspread(entries, ws: Worksheet, cell_range=None) -> None:
    if cell_range is None:
        # start from A1
        cell_range = "A1:"
        num_rows = len(entries)
        num_cols = max([len(row) for row in entries])
        cell_range += gspread.utils.rowcol_to_a1(num_rows, num_cols)
    ws.format(cell_range, {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    ws.update(cell_range, entries)
    # ws.update_title("[GID0]TestTitle")

    # Format example:
    # cells_list = ws.range(1, 1, num_rows, num_cols) # row, column, row_end, column_end. 1 1 stands for A1
    # cells_list = ws.range("E1:G120")
    # ws.format(cell_range, {"numberFormat": {"type": "DATE", "pattern": "mmmm dd"}, "horizontalAlignment": "CENTER"})


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    graphiler_dir_to_upload = find_latest_subdirectory("misc/artifacts", "graphiler_")
    print("Uploading results from", graphiler_dir_to_upload)
    graphiler_names_and_info = extract_graphiler_and_its_baselines_results_from_folder(
        graphiler_dir_to_upload
    )
    graphiler_worksheet_title = (
        f"[{socket.gethostname()}]{graphiler_dir_to_upload.split('/')[-1]}"
    )
    try:
        update_gspread(
            graphiler_names_and_info,
            create_worksheet(SPREADSHEET_URL, graphiler_worksheet_title)
            # open_worksheet(SPREADSHEET_URL, "0") # GID0 reserved for testing
        )
    except Exception as e:
        print("Failed to upload graphiler results:", e)

    dir_to_upload = find_latest_subdirectory("misc/artifacts", "benchmark_all_")
    print("Uploading results from", dir_to_upload)
    names_and_info = extract_het_results_from_folder(dir_to_upload)
    print(names_and_info)
    worksheet_title = f"[{socket.gethostname()}]{dir_to_upload.split('/')[-1]}"
    try:
        update_gspread(
            names_and_info,
            create_worksheet(SPREADSHEET_URL, worksheet_title)
            # open_worksheet(SPREADSHEET_URL, "0") # GID0 reserved for testing
        )
    except Exception as e:
        print("Failed to upload results:", e)
