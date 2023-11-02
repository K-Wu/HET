"""This script obtain the specified ncu sheet from the google spreadsheet, and use sass_analyzer roofline.py to plot the roofline. The results are normalized according to their actual peak memory roof and compute roof."""
from typing import Any
from .nsight_utils import (
    ask_subdirectory_or_file,
    get_pretty_hostname,
    get_worksheet_gid,
    open_worksheet,
    is_config_selected,
    ask_pretty_hostname,
)

import traceback

from .sass_analyzer.visualizer.templates import roofline as roofline
from .upload_benchmark_results import SPREADSHEET_URL
from .detect_pwd import is_pwd_het_dev_root, RESULTS_DIR


class CSVHeaderUtils:
    """This class provides static functions to extract information from the INFO columns."""

    MODEL_NAMES = {"HGT", "RGAT", "RGCN"}
    DATASET_NAMES = {
        "aifb",
        "am",
        "bgs",
        "biokg",
        "fb15k",
        "mag",
        "mutag",
        "wikikg2",
    }

    @staticmethod
    def get_info_column_idxes(csv_rows: list[list[str]]) -> list[int]:
        header = csv_rows[0]
        info_column_idxes: list[int] = []
        for idx, name in enumerate(header):
            if name.startswith("INFO"):
                info_column_idxes.append(idx)
        return info_column_idxes

    @staticmethod
    def get_column_name_idx_map(csv_rows: list[list[str]]) -> dict[str, int]:
        header = csv_rows[0]
        column_name_idx_map: dict[str, int] = {}
        for idx, name in enumerate(header):
            column_name_idx_map[name] = idx
        return column_name_idx_map

    @staticmethod
    def _get_model_or_dataset_column_idx(
        csv_rows: list[list[str]], column_value_set: set[str]
    ):
        info_column_idxes = CSVHeaderUtils.get_info_column_idxes(csv_rows)
        assert len(csv_rows) >= 2, (
            "There should be at least one row of data (other than header) in"
            " csv_rows"
        )

        row = csv_rows[1]
        for idx in info_column_idxes:
            if row[idx] in column_value_set:
                return idx
        assert False, "Cannot find model or dataset column"

    @staticmethod
    def get_model_column_idx(csv_rows: list[list[str]]) -> int:
        return CSVHeaderUtils._get_model_or_dataset_column_idx(
            csv_rows, CSVHeaderUtils.MODEL_NAMES
        )

    @staticmethod
    def get_dataset_column_idx(csv_rows: list[list[str]]) -> int:
        return CSVHeaderUtils._get_model_or_dataset_column_idx(
            csv_rows, CSVHeaderUtils.DATASET_NAMES
        )

    @staticmethod
    def get_dimension_column_idxes(csv_rows: list[list[str]]) -> list[int]:
        info_column_idxes = CSVHeaderUtils.get_info_column_idxes(csv_rows)
        assert len(csv_rows) >= 2, (
            "There should be at least one row of data (other than header) in"
            " csv_rows"
        )

        row = csv_rows[1]
        dimension_column_idxes: list[int] = []
        for idx in info_column_idxes:
            if row[idx].isdigit():
                dimension_column_idxes.append(idx)
        return dimension_column_idxes


class CSVIndexer:
    """This class indexes 1) the column name -> column index of the csv file, and
    2) row sets for each value of a given column name. An example of 2) is we may
    specify the model name as a dimension, and the indexer will create three sets,
    recording the row indexes of HGT, RGAT, and RGCN, respectively.
    We can then figure out the row indexes of chosen cell value combinations by set intersection.
    """

    def __init__(self, csv_rows: list[list[str]], indexed_columns: set[str]):
        self.csv_rows: list[list[str]] = csv_rows
        self.column_idx: dict[
            str, int
        ] = CSVHeaderUtils.get_column_name_idx_map(csv_rows)
        self.column_value_rows: dict[str, dict[str, set[int]]] = {}

        self.header: list[str] = csv_rows[0]
        self.dataset_column_idx: int = CSVHeaderUtils.get_dataset_column_idx(
            csv_rows
        )
        self.model_column_idx: int = CSVHeaderUtils.get_model_column_idx(
            csv_rows
        )
        self.dimension_column_idxes: list[
            int
        ] = CSVHeaderUtils.get_dimension_column_idxes(csv_rows)

        for column_name in indexed_columns:
            self.column_value_rows[column_name] = {}
            for row_idx, row in enumerate(csv_rows):
                column_value = row[self.column_idx[column_name]]
                if column_value not in self.column_value_rows[column_name]:
                    self.column_value_rows[column_name][column_value] = set()
                self.column_value_rows[column_name][column_value].add(row_idx)

    def get_models(self) -> set[str]:
        return set(
            self.column_value_rows[self.header[self.model_column_idx]].keys()
        )

    def get_datasets(self) -> set[str]:
        return set(
            self.column_value_rows[self.header[self.dataset_column_idx]].keys()
        )

    def get_dimensions(self) -> set[tuple[int, ...]]:
        dimensions: set[tuple[int, ...]] = set()
        for row in self.csv_rows[1:]:
            dimensions.add(
                tuple(
                    map(int, map(row.__getitem__, self.dimension_column_idxes))
                )
            )
        return dimensions

    def select_rows(self, column_values: dict[str, Any]) -> set[int]:
        """Select rows that satisfy the given column value constraints.
        The constraints are specified as a dictionary from column name to column value.
        """
        row_sets: list[set[int]] = []
        for column_name, column_value in column_values.items():
            row_sets.append(self.column_value_rows[column_name][column_value])

        return set.intersection(*row_sets)

    def get_column_idx(self, key: str) -> int:
        return self.column_idx[key]


if __name__ == "__main__":
    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"

    host_name = ask_pretty_hostname()
    path_name = ask_subdirectory_or_file(
        "misc/artifacts", "ncu_breakdown_", RESULTS_DIR
    )

    worksheet_title = f"[{host_name}]{path_name.split('/')[-1]}"[:100]
    try:
        worksheet_gid = get_worksheet_gid(SPREADSHEET_URL, worksheet_title)
        worksheet = open_worksheet(SPREADSHEET_URL, worksheet_gid)
        csv_rows: list[list[str]] = worksheet.get_all_values()
    except Exception as e:
        print("Failed to open worksheet:", e)
        print("Please run upload_cache_and_roofline.py first.")
        print(traceback.format_exc())
        exit(1)

    # The first row is the header
    # Grab "DRAM Roofline" (GB/s), "DRAM Achieved Traffic" (GB/s), "Compute Roofline" (GFLOP/s), "Achieved Work" (GFLOP/s), and  "Duration" (usecond) from ncu_ sheets
    #
    # The units are canonicalized in extract_ncu_values_from_details.py in hrt/utils/nsight_utils/load_nsight_report.py

    # INFO[0] -- INFO[N(7)-1] describes the experiment configuration
    # ID is the kernel launch id
    # "Pretty Name" is the kernel name
    # "Kernel Forward or Backward" is FwProp or BckProp
    # "Kernel Category" is GEMM or Traversal

    header = csv_rows[0]
    needed_column_idx: dict[str, int] = {
        "DRAM Roofline": header.index("DRAM Roofline"),
        "DRAM Achieved Traffic": header.index("DRAM Achieved Traffic"),
        "Compute Roofline": header.index("Compute Roofline"),
        "Achieved Work": header.index("Achieved Work"),
        # "Duration": header.index("Duration"), Draw duration graph separately using Tableau
        "Pretty Name": header.index("Pretty Name"),
        "Kernel Forward or Backward": header.index(
            "Kernel Forward or Backward"
        ),
        "Kernel Category": header.index("Kernel Category"),
    }

    info_column_idxes = CSVHeaderUtils.get_info_column_idxes(csv_rows)
    csv_rows: list[list[str]] = csv_rows[1:]
    selected_csv_rows: list[list[str]] = [csv_rows[0]]
    # Each row is a kernel launch stat
    for row in csv_rows:
        # Skip configurations that enable CompactDirect or Fusion
        if not is_config_selected(
            list(map(row.__getitem__, info_column_idxes)), ["None", "None"]
        ):
            continue
        selected_csv_rows.append(row)

    csv_indexer = CSVIndexer(selected_csv_rows, set(needed_column_idx.keys()))

    # Draw the roofline model
    # Plot GEMM FwProp, GEMM BwProp, Traversal FwProp, Traversal BwProp in four different colors
    # Draw a figure for each of the models or each of the hidden dimension sizes
    # TODO: finish this
    for model in csv_indexer.get_models():
        for dataset in csv_indexer.get_datasets():
            for dimensions in csv_indexer.get_dimensions():
                dimensions_: str = "_".join(map(str, dimensions))

                # Select rows that satisfy the given column value constraints.
                # The constraints are specified as a dictionary from column name to column value.
                column_values_list: list = [
                    {
                        "Kernel Forward or Backward": "FwProp",
                        "Kernel Category": "GEMM",
                    },
                    {
                        "Kernel Forward or Backward": "BckProp",
                        "Kernel Category": "GEMM",
                    },
                    {
                        "Kernel Forward or Backward": "FwProp",
                        "Kernel Category": "Traversal",
                    },
                    {
                        "Kernel Forward or Backward": "BckProp",
                        "Kernel Category": "Traversal",
                    },
                ]
                for column_values in column_values_list:
                    row_idxes = csv_indexer.select_rows(column_values)
                if len(row_idxes) == 0:
                    continue
