# Some code is from https://github.com/COVID19Tracking/ltc-data-processing
# And https://github.com/nlioc4/FSBot/blob/f7f1a000ec7d02056c136fe68b7f0ca2271c80ae/modules/accounts_handler.py#L326
# To create a credential, or set up a new spreasheet, follow instruction at https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem
from typing import Union, Any
from .detect_pwd import is_pwd_het_dev_root, RESULTS_RELATIVE_DIR

import os
import socket
import traceback

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


def count_rows(csv_rows: "list[list[Any]]") -> int:
    return len(csv_rows)


def count_cols(csv_rows: "list[list[Any]]") -> int:
    return max([len(row) for row in csv_rows])


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
    def validate_config_fmt(cls, input_fmt: str) -> None:
        configs = input_fmt.split(".")
        assert "ax_in" in configs
        assert "ax_out" in configs
        assert "ax_head" in configs

    @classmethod
    def canonicalize_list(cls, config: "list[str]", input_fmt: str) -> "list[str]":
        """
        Example of input_fmt: "flag_mul.flag_compact.ax_in.ax_out.ax_head"
        """
        cls.validate_config_fmt(input_fmt)
        if input_fmt is not None:
            config = cls.permute(input_fmt, config)
        return [c[2:] if c.startswith("--") else c for c in config]

    @classmethod
    def get_dimensions(cls, config: "list[str]", input_fmt: str) -> str:
        input_fmts = input_fmt.split(".")
        ax_in_idx = input_fmts.index("ax_in")
        ax_out_idx = input_fmts.index("ax_out")
        ax_head_idx = input_fmts.index("ax_head")
        return f"{config[ax_in_idx]}.{config[ax_out_idx]}.{config[ax_head_idx]}"

    @classmethod
    def get_configs_other_than_dimensions(
        cls, config: "list[str]", input_fmt: str
    ) -> str:
        config = cls.canonicalize_list(config, input_fmt)
        input_fmts = input_fmt.split(".")
        ax_in_idx = input_fmts.index("ax_in")
        ax_out_idx = input_fmts.index("ax_out")
        ax_head_idx = input_fmts.index("ax_head")
        other_configs: "list[str]" = [
            c
            for idx, c in enumerate(config)
            if idx not in {ax_in_idx, ax_out_idx, ax_head_idx}
        ]
        if max([len(c) for c in other_configs]) == 0:
            # Use $UNOPT to represent the unoptimized config
            return "$UNOPT"
        else:
            return ".".join(other_configs)

    @classmethod
    def to_str(cls, config: "list[str]", input_fmt: str) -> str:
        return ".".join(cls.canonicalize_list(config, input_fmt))


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
        return [model, dataset] + ConfigCanonicalizer.canonicalize_list(
            configs, config_fmt
        )

    @classmethod
    def to_str(cls, name: str, input_fmt: str) -> str:
        return ".".join(cls.to_list(name, input_fmt))


def extract_info_from(filename) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.result.log
    return NameCanonicalizer.to_list(
        filename[: filename.rfind(".result.log")],
        "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
    )


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
        if user_input.startswith("///"):  # user input is a relative path to het root
            assert user_input[3:].startswith(RESULTS_RELATIVE_DIR)
            user_input = os.path.relpath(user_input[3:], RESULTS_RELATIVE_DIR)
        result = os.path.join(root, user_input)
    assert os.path.exists(result), f"{result} does not exist"
    return result


def ask_subdirectory(root, prefix) -> str:
    while 1:
        result = ask_subdirectory_or_file(root, prefix)
        if os.path.isdir(result):
            return result
        else:
            print(result, "is not a directory. try again")
    raise RuntimeError("Unreachable")


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


def get_cell_range_from_A1(
    num_rows: int, num_cols: int, row_idx_beg: int = 0, col_idx_beg: int = 0
) -> str:
    """
    In future, we may use a1_range_to_grid_range to get the boundary of an existent worksheet.
    a1_range_to_grid_range returns (beg, end] for both row and column, i.e.,
    a1_range_to_grid_range('A1:A1')
    {'startRowIndex': 0, 'endRowIndex': 1, 'startColumnIndex': 0, 'endColumnIndex': 1}
    """
    # rowcol_to_a1(1,1) == 'A1'
    cell_range = gspread.utils.rowcol_to_a1(row_idx_beg + 1, col_idx_beg + 1)
    cell_range += ":"
    cell_range += gspread.utils.rowcol_to_a1(
        row_idx_beg + num_rows, col_idx_beg + num_cols
    )
    print(cell_range)
    return cell_range


def try_best_to_numeric(
    csv_rows: "list[list[Union[float, str, int]]]",
) -> "list[list[Union[float, str, int]]]":
    new_csv_rows: "list[list[Union[float, str, int]]]" = []
    for row in csv_rows:
        new_row = []
        for ele in row:
            if isinstance(ele, str) and ele.isnumeric():
                new_row.append(int(ele))
            elif isinstance(ele, str) and ele.replace(".", "", 1).isnumeric():
                new_row.append(float(ele))
            else:
                new_row.append(ele)
        new_csv_rows.append(new_row)
    return new_csv_rows


def update_gspread(entries, ws: Worksheet, cell_range=None) -> None:
    if cell_range is None:
        # start from A1
        num_rows = len(entries)
        num_cols = max([len(row) for row in entries])
        cell_range = get_cell_range_from_A1(num_rows, num_cols)
    ws.format(cell_range, {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    ws.update(cell_range, try_best_to_numeric(entries))
    # ws.update_title("[GID0]TestTitle")

    # Format example:
    # cells_list = ws.range(1, 1, num_rows, num_cols) # row, column, row_end, column_end. 1 1 stands for A1
    # cells_list = ws.range("E1:G120")
    # ws.format(cell_range, {"numberFormat": {"type": "DATE", "pattern": "mmmm dd"}, "horizontalAlignment": "CENTER"})


def upload_folder(
    root: str, prefix: str, is_graphiler_flag: bool, test_repeat_x_y: bool = False
):
    dir_to_upload = find_latest_subdirectory(root, prefix)
    print("Uploading results from", dir_to_upload)
    if is_graphiler_flag:
        names_and_info = extract_graphiler_and_its_baselines_results_from_folder(
            dir_to_upload
        )
    else:
        names_and_info = extract_het_results_from_folder(dir_to_upload)
    print(names_and_info)
    worksheet_title = f"[{socket.gethostname()}]{dir_to_upload.split('/')[-1]}"
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
                    count_rows(names_and_info), count_cols(names_and_info), 0, 0
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
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    upload_folder("misc/artifacts", "graphiler_", True, False)
    upload_folder("misc/artifacts", "benchmark_all_", False, False)
