# Some code is from https://github.com/COVID19Tracking/ltc-data-processing
# And https://github.com/nlioc4/FSBot/blob/f7f1a000ec7d02056c136fe68b7f0ca2271c80ae/modules/accounts_handler.py#L326
# To create a credential, or set up a new spreasheet, follow instruction at https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem

import os

# TODO: find and extract the following pattern from the result folder
#
# "Mean forward time: {:4f} ms"
# "Mean backward time: {:4f} ms"
# "Mean training time: {:4f} ms"
# OUTPUT_DIR="misc/artifacts/benchmark_all_`date +%Y%m%d%H%M`"
#

SPREADSHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1qMklewOvYRVRHTYlMErvyd67afJvaVNwd79sMrKw__4/",
)
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


def extract_info_from(filename):
    # model_name.dataset_name.mul_flag.compact_flag.result.log
    return filename.split(".")[:-2]


def find_latest_subdirectory(root, prefix):
    candidates = []
    for subdir in os.listdir(root):
        if subdir.startswith(prefix):
            candidates.append(subdir)
    return os.path.join(root, max(candidates))


def extract_results_from_folder(path):
    all_names_and_info = []
    for filename in os.listdir(path):
        if filename.endswith(".result.log"):
            name_info = extract_info_from(filename)
            with open(os.path.join(path, filename), "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("Mean forward time"):
                        name_info.append(float(line.split(":")[1].strip().split()[0]))
                    elif line.startswith("Mean backward time"):
                        name_info.append(float(line.split(":")[1].strip().split()[0]))
                    elif line.startswith("Mean training time"):
                        name_info.append(float(line.split(":")[1].strip().split()[0]))
            all_names_and_info.append(name_info)
    return all_names_and_info


def update_gspread(entries, target_sheet_url, target_gid, cell_range=None):
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

    if cell_range is None:
        # start from A1
        cell_range = "A1:"
        num_rows = len(entries)
        num_cols = max([len(row) for row in entries])
        cell_range += gspread.utils.rowcol_to_a1(num_rows, num_cols)
    ws.format(cell_range, {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}})
    ws.update(cell_range, entries)
    ws.update_title("[GID0]TestTitle")
    # Format example:
    # cells_list = ws.range(1, 1, num_rows, num_cols) # row, column, row_end, column_end. 1 1 stands for A1
    # cells_list = ws.range("E1:G120")
    # ws.format(cell_range, {"numberFormat": {"type": "DATE", "pattern": "mmmm dd"}, "horizontalAlignment": "CENTER"})


if __name__ == "__main__":
    dir_to_upload = find_latest_subdirectory("misc/artifacts", "benchmark_all_")
    print("Uploading results from", dir_to_upload)
    names_and_info = extract_results_from_folder(dir_to_upload)
    print(names_and_info)
    update_gspread(
        names_and_info,
        SPREADSHEET_URL,
        "0",  # GID0 reserved for testing
    )
