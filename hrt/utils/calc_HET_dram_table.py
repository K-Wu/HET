from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import (
    SPREADSHEET_URL,
    extract_het_results_from_folder,
)
from .nsight_utils import (
    ask_subdirectory,
    update_gspread,
    create_worksheet,
    get_cell_range_from_A1,
    count_cols,
    count_rows,
    get_pretty_hostname,
)
from .calc_tables import BenchAllRecords, calc_best_HET_and_show_all
from .detect_pwd import RESULTS_DIR
import traceback

if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    het_results_dir = ask_subdirectory(
        "misc/artifacts", "benchmark_64_", RESULTS_DIR
    )
    print("Obtaining results from", het_results_dir)
    het_names_and_info = extract_het_results_from_folder(het_results_dir)
    (
        all_het_time_records,
        all_het_dram_records,
        all_het_status_records,
    ) = BenchAllRecords.load_HET_results_from_uploader(
        het_names_and_info, "flag_mul.flag_compact.ax_in.ax_out.ax_head"
    )

    # Draw tables
    tab_best_het_dram = calc_best_HET_and_show_all(
        all_het_dram_records, all_het_status_records, 64, 64, 1
    )

    tab_best_het_time = calc_best_HET_and_show_all(
        all_het_time_records, all_het_status_records, 64, 64, 1
    )

    # upload it
    worksheet_title = (
        f"[{get_pretty_hostname()}]dram_table.{het_results_dir.split('/')[-1]}"
    )
    try:
        worksheet = create_worksheet(SPREADSHEET_URL, worksheet_title)
        # Increase the size of grid if needed
        default_rows = 100
        default_cols = 26
        total_num_rows = max(
            [
                count_rows(tab_best_het_dram),
                count_rows(tab_best_het_time),
            ]
        )
        total_num_cols = count_cols(tab_best_het_dram) + count_cols(
            tab_best_het_time
        )
        if total_num_rows > default_rows or total_num_cols > default_cols:
            worksheet.resize(total_num_rows, total_num_cols)
        update_gspread(
            tab_best_het_dram,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het_dram),
                count_cols(tab_best_het_dram),
                0,
                0,
            ),
        )
        update_gspread(
            tab_best_het_time,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het_time),
                count_cols(tab_best_het_time),
                0,
                count_cols(tab_best_het_dram),
            ),
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
