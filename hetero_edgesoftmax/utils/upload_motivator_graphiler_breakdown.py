from .detect_pwd import is_pwd_het_dev_root
from .load_nsight_report import load_nsys_report
from .upload_benchmark_results import (
    ask_subdirectory_or_file,
    create_worksheet,
    update_gspread,
    SPREADSHEET_URL,
    get_pretty_hostname,
)
from .upload_cache_and_roofline import extract_info_from_nsys
import traceback
import os


def simple_combine_nsys_csvs(raw_csvs: list[list[list[str]]]) -> "list[list[str]]":
    """
    This function asserts headers are the same for all csvs, and keep only one header and merge the bodies of all csvs.
    """
    assert len(raw_csvs) > 0, "raw_csvs must not be empty"
    header = raw_csvs[0][0]
    for csv in raw_csvs:
        assert csv[0] == header, "Headers must be the same for all csvs"
    return [header] + [item for sublist in raw_csvs for item in sublist[1:]]


def upload_nsys_report(subdir_path: str, nsys_report_name: str):
    raw_csvs: list[list[list[str]]] = []
    for filename in os.listdir(subdir_path):
        print("extract Processing", filename)
        if filename.endswith(".nsys-rep"):
            curr_csv: list[list[str]] = load_nsys_report(
                os.path.join(subdir_path, filename), nsys_report_name
            )
            info_from_filename: list[str] = extract_info_from_nsys(filename)
            curr_csv = [info_from_filename + row for row in curr_csv]
            # For info from filename, Set INFO[idx] as the column names in header row
            for idx_col in range(len(info_from_filename)):
                curr_csv[0][idx_col] = f"INFO[{idx_col}]"
            raw_csvs.append(curr_csv)

    # Combine all csvs into one
    # csv_rows = [item for sublist in raw_csvs for item in sublist]
    csv_rows = simple_combine_nsys_csvs(raw_csvs)
    print(csv_rows)
    # Create worksheet
    worksheet_title = f"[{get_pretty_hostname()}]{subdir_path.split('/')[-1]}"[:100]
    try:
        worksheet = create_worksheet(SPREADSHEET_URL, worksheet_title)
    except Exception as e:
        print("Failed to create worksheet:", e)
        print(traceback.format_exc())
        exit(1)

    # Upload
    try:
        update_gspread(csv_rows, worksheet)
    except Exception as e:
        print("Failed to upload ncu results:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    path_name = ask_subdirectory_or_file(
        "misc/artifacts", "motivator_graphiler_breakdown_"
    )
    upload_nsys_report(path_name, "cuda_kern_exec_sum")
