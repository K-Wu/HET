from .upload_benchmark_results import (
    extract_het_results_from_folder,
    SPREADSHEET_URL,
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
from .detect_pwd import RESULTS_DIR, is_pwd_het_dev_root
import traceback
from .calc_tables import (
    BenchAllRecords,
    calc_best_HET_and_show_all,
    calc_opt_matrix,
)

if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    # Load data from the results folder
    het_results_dir = ask_subdirectory(
        "misc/artifacts", "benchmark_all_", RESULTS_DIR
    )
    print("Obtaining results from", het_results_dir)
    het_names_and_info = extract_het_results_from_folder(het_results_dir)
    (
        all_het_time_records,
        all_het_status_records,
    ) = BenchAllRecords.load_HET_results_from_uploader(
        het_names_and_info, "flag_mul.flag_compact.ax_in.ax_out.ax_head"
    )

    # Draw tables
    tab_best_het = calc_best_HET_and_show_all(
        all_het_time_records, all_het_status_records, 64, 64, 1
    )
    tab_opt_matrix = calc_opt_matrix(all_het_time_records, 64, 64, 1)

    # upload it
    worksheet_title = f"[{get_pretty_hostname()}]tables.sweep.{het_results_dir.split('/')[-1]}"
    try:
        worksheet = create_worksheet(SPREADSHEET_URL, worksheet_title)

        # Increase the size of grid if needed
        default_rows = 100
        default_cols = 26
        total_num_rows = max(
            [
                count_rows(tab_best_het),
                count_rows(tab_opt_matrix),
            ]
        )
        total_num_cols = +count_cols(tab_best_het) + count_cols(tab_opt_matrix)
        if 3 * total_num_rows > default_rows or total_num_cols > default_cols:
            worksheet.resize(3 * total_num_rows, total_num_cols)
        update_gspread(
            tab_best_het,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het), count_cols(tab_best_het), 0, 0
            ),
        )
        update_gspread(
            tab_opt_matrix,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_opt_matrix),
                count_cols(tab_opt_matrix),
                0,
                +count_cols(tab_best_het),
            ),
        )

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        exit()

    # print het 32,32 and 128,128 tables
    try:
        tab_best_het_32 = calc_best_HET_and_show_all(
            all_het_time_records, all_het_status_records, 32, 32, 1
        )
        tab_opt_matrix_32 = calc_opt_matrix(all_het_time_records, 32, 32, 1)
        update_gspread(
            tab_best_het_32,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het_32),
                count_cols(tab_best_het_32),
                total_num_rows,
                0,
            ),
        )
        update_gspread(
            tab_opt_matrix_32,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_opt_matrix_32),
                count_cols(tab_opt_matrix_32),
                total_num_rows,
                count_cols(tab_best_het),
            ),
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())

    try:
        tab_best_het_128 = calc_best_HET_and_show_all(
            all_het_time_records, all_het_status_records, 128, 128, 1
        )
        tab_opt_matrix_128 = calc_opt_matrix(all_het_time_records, 128, 128, 1)
        update_gspread(
            tab_best_het_128,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het_128),
                count_cols(tab_best_het_128),
                2 * total_num_rows,
                0,
            ),
        )
        update_gspread(
            tab_opt_matrix_128,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_opt_matrix_128),
                count_cols(tab_opt_matrix_128),
                2 * total_num_rows,
                count_cols(tab_best_het),
            ),
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
