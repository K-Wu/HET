# TODO: add an arithmetic intensity column to the gspread sheet to allow tableau roofline plot
# TODO: load ncu results again, upload it, and append the theoretical occupancy and arithmetic intensity columns
from .nsight_utils import (
    get_worksheet_gid,
    open_worksheet,
    get_cell_range_from_A1,
)
import gspread
from .upload_benchmark_results import SPREADSHEET_URL
from .plot_roofline import CSVHeaderUtils

import logging

LOG = logging.getLogger(__name__)

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d]"
        " %(threadName)15s: %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def get_primary_keys_to_row_idx_map(
    csv_rows: list[list[str]], primary_keys: list[str]
) -> dict[tuple[str, ...], int]:
    column_name_idx_map: dict[
        str, int
    ] = CSVHeaderUtils.get_column_name_idx_map(csv_rows)
    primary_key_idxes: list[int] = [
        column_name_idx_map[key] for key in primary_keys
    ]
    primary_keys_to_row_idx_map: dict[tuple[str, ...], int] = {}
    for row_idx, row in enumerate(csv_rows[1:]):
        primary_key = tuple(row[idx] for idx in primary_key_idxes)
        primary_keys_to_row_idx_map[primary_key] = row_idx
    return primary_keys_to_row_idx_map


def get_columns_to_append(
    dst_sheet: list[list[str]],
    src_sheet: list[list[str]],
    new_column_names: list[str],
    primary_keys: list[str],
) -> tuple[int, list[list[str]]]:
    """Return the new column idx (beginning index if there are multiple columns) and column content, i.e., cell range and cell values to be passed to gspread update_worksheet.
    dst_sheet is the sheet to be updated. src_sheet is the sheet which supplies the column with column_name. primary_keys is a list of column names that uniquely identify a row record.
    """
    src_primary_keys_to_row: dict[
        tuple[str, ...], int
    ] = get_primary_keys_to_row_idx_map(src_sheet, primary_keys)
    src_column_idx: dict[str, int] = CSVHeaderUtils.get_column_name_idx_map(
        src_sheet
    )
    dst_column_idx: dict[str, int] = CSVHeaderUtils.get_column_name_idx_map(
        dst_sheet
    )
    dst_primary_key_idxes: list[int] = [
        dst_column_idx[key] for key in primary_keys
    ]
    src_new_columns_idxes: list[int] = [
        src_column_idx[key] for key in new_column_names
    ]

    # Note that in ncu sheet, there are two header rows, the name row and the unit row.
    # The following scheme still works because the second row, now considered as a data row, has the unique primary keys ("", "", ...)
    new_columns: list[list[str]] = [  # Header
        list(map(src_sheet[0].__getitem__, src_new_columns_idxes))
    ]
    for row in dst_sheet[1:]:
        curr_primary_keys: tuple[str, ...] = tuple(
            row[idx] for idx in dst_primary_key_idxes
        )
        try:
            src_row_idx = src_primary_keys_to_row[curr_primary_keys]
            new_columns.append(
                list(
                    map(
                        src_sheet[src_row_idx].__getitem__,
                        src_new_columns_idxes,
                    )
                )
            )
        except:
            LOG.warning(f"{curr_primary_keys} not found in src_sheet")
            new_columns.append([""] * len(new_column_names))

    return len(dst_sheet[0]), new_columns


def append_columns(
    dst_worksheet: gspread.Worksheet,
    src_worksheet: gspread.Worksheet,
    new_column_names: list[str],
    primary_keys: list[str],
):
    """dst_sheet is the sheet to be updated. src_sheet is the sheet which supplies the column with column_name. primary_keys is a list of column names that uniquely identify a row record."""
    # Read the dst_sheet and src_sheet into csv_rows
    dst_csv_rows: list[list[str]] = dst_worksheet.get_all_values()
    src_csv_rows: list[list[str]] = src_worksheet.get_all_values()
    # Get the new column idx and column content
    new_column_idx, new_column_content = get_columns_to_append(
        dst_csv_rows, src_csv_rows, new_column_names, primary_keys
    )
    # Update the dst_sheet
    cell_range = get_cell_range_from_A1(
        len(new_column_content), len(new_column_content[0]), 0, new_column_idx
    )
    LOG.info(f"Updating {cell_range}")
    if dst_worksheet.row_count < len(new_column_content):
        LOG.info(
            "Exceeding current grid limit. Resizing row count from"
            f" {dst_worksheet.row_count} to {len(new_column_content)}"
        )
        dst_worksheet.resize(len(new_column_content), None)
    if dst_worksheet.col_count < new_column_idx + len(new_column_names):
        LOG.info(
            "Exceeding current grid limit. Resizing column count from"
            f" {dst_worksheet.col_count} to"
            f" {new_column_idx + len(new_column_names)}"
        )
        dst_worksheet.resize(None, new_column_idx + len(new_column_names))
    dst_worksheet.format(
        cell_range, {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}}
    )
    dst_worksheet.update(
        f"{cell_range}",
        new_column_content,
    )


if __name__ == "__main__":
    # sheet name: ASPLOS[kwu-csl227-99]ncu_selected_202307180518
    # name of sheet to provide new columns: [kwu-csl227-99]ncu_selected_202307180518
    src_sheet_gid = get_worksheet_gid(
        SPREADSHEET_URL, "[kwu-csl227-99]ncu_selected_202307180518"
    )
    src_worksheet = open_worksheet(
        SPREADSHEET_URL, src_sheet_gid, assert_gid_is_zero=False
    )
    dst_sheet_gid = get_worksheet_gid(
        SPREADSHEET_URL, "ASPLOS[kwu-csl227-99]ncu_selected_202307180518"
    )
    dst_worksheet = open_worksheet(
        SPREADSHEET_URL, dst_sheet_gid, assert_gid_is_zero=False
    )
    append_columns(
        dst_worksheet,
        src_worksheet,
        ["Theoretical Occupancy", "Arithmetic Intensity"],
        [
            "INFO[0]",
            "INFO[1]",
            "INFO[2]",
            "INFO[3]",
            "INFO[4]",
            "INFO[5]",
            "INFO[6]",
            "ID",
        ],
    )
