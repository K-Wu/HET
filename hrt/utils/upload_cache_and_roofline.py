from .upload_benchmark_results import SPREADSHEET_URL
from .nsight_utils import (
    ask_subdirectory_or_file,
    update_gspread,
    create_worksheet,
    get_pretty_hostname,
)
from .nsight_utils import (
    extract_from_ncu_folder,
    extract_from_ncu_file,
)
from .classify_het_kernels import classify_het_kernel
from .detect_pwd import is_pwd_het_dev_root, RESULTS_RELATIVE_DIR
import traceback
import os

if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    path_name = ask_subdirectory_or_file(
        "misc/artifacts", "ncu_breakdown_", RESULTS_RELATIVE_DIR
    )

    # Create worksheet
    worksheet_title = f"[{get_pretty_hostname()}]{path_name.split('/')[-1]}"[
        :100
    ]
    try:
        worksheet = create_worksheet(
            SPREADSHEET_URL, worksheet_title, retry=True
        )
    except Exception as e:
        print("Failed to create worksheet:", e)
        print(traceback.format_exc())
        exit(1)

    # Extract results
    if os.path.isdir(path_name):
        csv_rows = extract_from_ncu_folder(
            path_name, True, True, classify_het_kernel
        )
    else:
        csv_rows = extract_from_ncu_file(
            path_name, True, True, classify_het_kernel
        )

    # Upload
    try:
        update_gspread(csv_rows, worksheet)
    except Exception as e:
        print("Failed to upload ncu results:", e)
        print(traceback.format_exc())
