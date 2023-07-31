from .detect_pwd import is_pwd_het_dev_root
from .load_nsight_report import load_nsys_report
from .upload_benchmark_results import (
    ask_subdirectory_or_file,
    create_worksheet,
    update_gspread,
    SPREADSHEET_URL,
    get_pretty_hostname,
)
import traceback
import os

if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    path_name = ask_subdirectory_or_file(
        "misc/artifacts", "motivator_graphiler_breakdown_"
    )
    raw_csvs: list[list[list[str]]] = []
    len_info_from_filename: int = -1
    for filename in os.listdir(path_name):
        print("extract Processing", filename)
        if filename.endswith(".nsys-rep"):
            curr_csv: list[list[str]] = load_nsys_report(
                os.path.join(path_name, filename), "cuda_kern_exec_sum"
            )
            curr_csv = [[filename] + row for row in curr_csv]
            raw_csvs.append(curr_csv)
    csv_rows = [item for sublist in raw_csvs for item in sublist]

    # Create worksheet
    worksheet_title = f"[{get_pretty_hostname()}]{path_name.split('/')[-1]}"[:100]
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
