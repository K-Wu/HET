# some code is from https://github.com/COVID19Tracking/ltc-data-processing
# To create a credential, or set up a new spreasheet, follow instruction at https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem


def upload_benchmark_results(cell_range, entries, target_sheet_url, target_gid):
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

    ws.update(cell_range, entries)


if __name__ == "__main__":
    upload_benchmark_results(
        "A1:C3",
        [
            ["test_header", "test_header_1", "test_header_2"],
            ["2020-01-01", "test", "1.0"],
            ["2020-01-02", "test2", "2.0"],
        ],
        "https://docs.google.com/spreadsheets/d/1qMklewOvYRVRHTYlMErvyd67afJvaVNwd79sMrKw__4/",
        "0",
    )
