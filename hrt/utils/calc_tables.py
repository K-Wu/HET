"""
This file contains table-making logic, mostly reproducing our tables in the Jan 2023 submission.
"""
from typing import Union, Callable
from .detect_pwd import is_pwd_het_dev_root
from .upload_benchmark_results import (
    extract_graphiler_and_its_baselines_results_from_folder,
    extract_het_results_from_folder,
    SPREADSHEET_URL,
)
from .nsight_utils import (
    ConfigCanonicalizer,
    ask_subdirectory,
    update_gspread,
    create_worksheet,
    get_cell_range_from_A1,
    count_cols,
    count_rows,
    get_pretty_hostname,
)
from .detect_pwd import RESULTS_DIR
import numpy as np
import traceback


class BenchAllRecords:
    # [inference/training][dataset][model][config]
    # for HET, config is canonicalized string and ConfigCanonicalizer should provide the necesary apis.
    # for baseline system, config is system name and requires function to figure out its system, i.e., graphiler, DGL, PyG, etc.

    all_records: "dict[str, dict[str, dict[str, dict[str, str]]]]"
    cfg_canonicalizer: Callable[["list[str]"], str]
    get_dimensions_from_cfg: Callable[["str"], "str"]
    get_rest_from_cfg: Callable[["str"], "str"]

    def __init__(self, cfg_canonicalizer: Callable[["list[str]"], str]):
        self.cfg_canonicalizer = cfg_canonicalizer
        self.all_records = {}

    def get_all_dataset(self) -> "set[str]":
        return set(self.all_records["inference"].keys()).union(
            set(self.all_records["training"].keys())
        )

    def datasets_in(self, mode: str) -> "set[str]":
        assert mode in ["inference", "training"]
        return set(self.all_records[mode].keys())

    def models_in(self, mode: str, dataset: str) -> "set[str]":
        assert mode in ["inference", "training"]
        assert dataset in self.all_records[mode]
        return set(self.all_records[mode][dataset].keys())

    def configs_in(self, mode: str, dataset: str, model: str) -> "set[str]":
        assert mode in ["inference", "training"]
        assert dataset in self.all_records[mode]
        assert model in self.all_records[mode][dataset]
        return set(self.all_records[mode][dataset][model].keys())

    def get_all_model(self) -> "set[str]":
        result: set[str] = set()
        for mode in ["inference", "training"]:
            for dataset in self.all_records[mode]:
                result = result.union(
                    set(self.all_records[mode][dataset].keys())
                )
        return result

    def get_all_config(
        self, models_filter: Union["set[str]", None] = None
    ) -> "set[str]":
        result: set[str] = set()
        for mode in ["inference", "training"]:
            for dataset in self.all_records[mode]:
                for model in self.all_records[mode][dataset]:
                    # skip if model is not in the filter
                    if (
                        models_filter is not None
                        and model not in models_filter
                    ):
                        continue
                    result = result.union(
                        set(self.all_records[mode][dataset][model].keys())
                    )
        return result

    def get_record(
        self, mode: str, dataset: str, model: str, config: str
    ) -> str:
        assert mode in ["inference", "training"]
        if (
            mode not in self.all_records
            or dataset not in self.all_records[mode]
            or model not in self.all_records[mode][dataset]
            or config not in self.all_records[mode][dataset][model]
        ):
            return "Not Presented"
        return self.all_records[mode][dataset][model][config]

    def store_record(
        self, mode: str, dataset: str, model: str, config: str, value: str
    ) -> None:
        assert mode in ["inference", "training"]
        if mode not in self.all_records:
            self.all_records[mode] = {}
        if dataset not in self.all_records[mode]:
            self.all_records[mode][dataset] = {}
        if model not in self.all_records[mode][dataset]:
            self.all_records[mode][dataset][model] = {}
        if config not in self.all_records[mode][dataset][model]:
            self.all_records[mode][dataset][model][config] = value

    @classmethod
    def get_HETAllRecords(cls, cfg_fmt: str) -> "BenchAllRecords":
        def cfg_canonicalizer(cfg: "list[str]") -> str:
            return ConfigCanonicalizer.to_str(cfg, cfg_fmt)

        return cls(cfg_canonicalizer)

    # TODO: support in_dim, out_dim, num_heads as parameters in the future. Right now we only collect 64.64.1
    @classmethod
    def load_baseline_results_from_uploader(
        cls, out_csv: "list[list[Union[float, str, int]]]"
    ) -> "dict[str, BenchAllRecords]":
        """
        This function takes in the csv from extract_graphiler_and_its_baselines_results_from_folder and store all the records in the result object
        Each row is in the format of
        model,	graphiler/baseline,dataset,	config_name, training/inference, time
        """
        time_records = dict()
        time_records["64.64.1"] = cls.get_BaselineAllRecords()
        for row in out_csv:
            row = list(map(str, row))
            model, _, dataset, cfg, mode, time = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
            )
            system = cls.get_base_system(cfg)
            time_records["64.64.1"].store_record(
                mode, dataset, model, system, time
            )
        return time_records

    @classmethod
    def load_HET_results_from_uploader(
        cls, out_csv: "list[list[Union[float, str, int]]]", cfg_fmt: str
    ) -> "tuple[dict[str, BenchAllRecords], dict[str, BenchAllRecords], dict[str, BenchAllRecords]]":
        """
        This function takes in the csv from extract_het_results_from_folder and store all the records in the result object
        cfg_format corresponds to the one specified in extract_info_from(filename) in upload_benchmark_results.py
        Each row is in the format of
        Model,	Dataset,    Config0,    Config1,    ...,	Inference Time, Backward Propagation Time,		Training Time,    Status
        """
        time_records: dict[
            str, BenchAllRecords
        ] = dict()  # cls.get_HETAllRecords(cfg_fmt)
        dram_records: dict[
            str, BenchAllRecords
        ] = dict()  # cls.get_HETAllRecords(cfg_fmt)
        status_records: dict[
            str, BenchAllRecords
        ] = dict()  # cls.get_HETAllRecords(cfg_fmt)
        for row in out_csv:
            row = list(map(str, row))
            model, dataset = row[0], row[1]
            # row[-6], row[-5], row[-4], row[-3], row[-2], row[-1] are
            # inference_time, backward_prop_time, training_time, status, inference_dram, training_dram
            inference_time = row[-6]
            # Omitting row[-5], which is backward prop time
            training_time = row[-4]
            status = row[-3]
            inference_dram = row[-2]
            training_dram = row[-1]
            configs = list(map(str, row[2:-6]))
            configs_dimensions = ConfigCanonicalizer.get_dimensions(
                configs, cfg_fmt
            )
            configs_rest: str = (
                ConfigCanonicalizer.get_configs_other_than_dimensions(
                    configs, cfg_fmt
                )
            )
            if configs_dimensions not in time_records:
                time_records[configs_dimensions] = cls.get_HETAllRecords(
                    cfg_fmt
                )
                status_records[configs_dimensions] = cls.get_HETAllRecords(
                    cfg_fmt
                )
                dram_records[configs_dimensions] = cls.get_HETAllRecords(
                    cfg_fmt
                )
            time_records[configs_dimensions].store_record(
                "inference",
                dataset,
                model,
                configs_rest,
                inference_time,
            )
            time_records[configs_dimensions].store_record(
                "training",
                dataset,
                model,
                configs_rest,
                training_time,
            )
            dram_records[configs_dimensions].store_record(
                "inference",
                dataset,
                model,
                configs_rest,
                inference_dram,
            )
            dram_records[configs_dimensions].store_record(
                "training",
                dataset,
                model,
                configs_rest,
                training_dram,
            )
            status_records[configs_dimensions].store_record(
                "inference",
                dataset,
                model,
                configs_rest,
                status,
            )
            status_records[configs_dimensions].store_record(
                "training",
                dataset,
                model,
                configs_rest,
                status,
            )
        return time_records, dram_records, status_records

    @classmethod
    def get_BaselineAllRecords(cls) -> "BenchAllRecords":
        def cfg_canonicalizer(cfg: "list[str]") -> str:
            return ".".join(cfg)

        return cls(cfg_canonicalizer)

    @classmethod
    def get_base_system(cls, cfg: str) -> str:
        """
        For use of BaselineAllRecords only, not HETAllRecords.
        """
        # TODO: collect Seastar and HGL
        BASELINE_SYSTEMS = {"DGL", "PyG", "Graphiler"}
        for candidate in BASELINE_SYSTEMS:
            if candidate in cfg:
                return candidate
        raise ValueError(
            f"Cannot find baseline system in {cfg}. Either this is a HET"
            " benchmark, or the config is not in the expected format."
        )


def is_float(input: str) -> bool:
    try:
        float(input)
        return True
    except ValueError:
        return False


def _calc_best(
    time_records: "dict[str, BenchAllRecords]",
    dimensions_cfg: str,  # e.g. "64.64.1"
    status_records: Union["dict[str, BenchAllRecords]", None] = None,
    flag_store_to_input_records: bool = True,
) -> "list[list[str]]":
    tmp_records: "dict[str, dict[str, dict[str, dict[str, str]]]]" = (
        {}
    )  # [mode][model][config][dataset]
    # Iterate through [inference/training], [dataset], [model], [config] in order
    for mode in ["inference", "training"]:
        for dataset in time_records[dimensions_cfg].datasets_in(mode):
            for model in time_records[dimensions_cfg].models_in(mode, dataset):
                # keep track of the best time and config
                best_time = float("inf")
                best_config = "$UNDEFINED"
                for config in time_records[dimensions_cfg].configs_in(
                    mode, dataset, model
                ):
                    time = time_records[dimensions_cfg].get_record(
                        mode, dataset, model, config
                    )
                    if mode not in tmp_records:
                        tmp_records[mode] = {}
                    if model not in tmp_records[mode]:
                        tmp_records[mode][model] = {}
                    if config not in tmp_records[mode][model]:
                        tmp_records[mode][model][config] = {}
                    if (
                        status_records is not None
                        and status_records[dimensions_cfg].get_record(
                            mode, dataset, model, config
                        )
                        != "OK"
                    ):
                        tmp_records[mode][model][config][
                            dataset
                        ] = status_records[dimensions_cfg].get_record(
                            mode, dataset, model, config
                        )
                    tmp_records[mode][model][config][dataset] = time
                    if is_float(time) and float(time) < best_time:
                        best_time = float(time)
                        best_config = config
                if "$BEST" not in tmp_records[mode][model]:
                    tmp_records[mode][model]["$BEST"] = {}
                    tmp_records[mode][model]["$BESTCONFIG"] = {}
                tmp_records[mode][model]["$BEST"][dataset] = str(best_time)
                tmp_records[mode][model]["$BESTCONFIG"][dataset] = best_config
                if flag_store_to_input_records:
                    time_records[dimensions_cfg].store_record(
                        mode, dataset, model, "$BEST", str(best_time)
                    )
                    time_records[dimensions_cfg].store_record(
                        mode, dataset, model, "$BESTCONFIG", best_config
                    )

    # Now print
    results: "list[list[str]]" = []
    for mode in ["inference", "training"]:
        # Keep the order same as the excel used in previous submission
        for model in sorted(tmp_records[mode], reverse=True):
            # Title of the sub-table
            results += [[mode, model]]
            datasets_: list[str] = sorted(
                time_records[dimensions_cfg].get_all_dataset(), reverse=True
            )
            # Header of the sub-table
            results.append(["system"] + datasets_)
            # put $BEST at the end
            for config in sorted(
                list(
                    set(tmp_records[mode][model]).difference(
                        {"$BEST", "$BESTCONFIG"}
                    )
                )
            ) + ["$BEST"]:
                row = [config]
                for dataset in datasets_:
                    if dataset not in tmp_records[mode][model][config]:
                        row.append("Not Presented")
                    else:
                        row.append(tmp_records[mode][model][config][dataset])
                results.append(row)
            results.append([])
    return results


# TODO: support in_dim, out_dim, num_heads as parameters in the future. Right now we only collect 64.64.1
def calc_best_baselines_and_show_all(
    all_time_records_per_dimension_cfg: "dict[str, BenchAllRecords]",
    in_dim: int,
    out_dim: int,
    num_heads: int,
) -> "list[list[str]]":
    """
    If not done, first obtain BenchAllRecords by

    baseline_records = BenchAllRecords.load_baseline_results_from_uploader(
        all_baseline_csv
    )

    An example output:
                          Inference
              aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    DGLa      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    DGLb      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    PyG       x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    Graphiler x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BEST      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
                          Training
              aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    DGLa      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    DGLb      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    PyG       x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    Seastar   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BEST      x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    """

    assert in_dim == 64, "Only support 64.64.1 for now"
    assert out_dim == 64, "Only support 64.64.1 for now"
    assert num_heads == 1, "Only support 64.64.1 for now"
    dimensions_cfg: str = ConfigCanonicalizer.get_dimensions(
        [str(in_dim), str(out_dim), str(num_heads)], "ax_in.ax_out.ax_head"
    )
    return _calc_best(all_time_records_per_dimension_cfg, dimensions_cfg)


def calc_best_HET_and_show_all(
    all_time_records_per_dimension_cfg: "dict[str, BenchAllRecords]",
    all_status_records_per_dimension_cfg: "dict[str, BenchAllRecords]",
    in_dim: int,
    out_dim: int,
    num_heads: int,
) -> "list[list[str]]":
    """
    If not done, first obtain BenchAllRecords by
    records = BenchAllRecords.load_HET_results_from_uploader(all_hector_csv, cfg_fmt)

    An example output:
                          Inference
        aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    U   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    F   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C+F x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BST x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
                          Training
        aifb  mutag	bgs   am	  mag	  wikikg2	fb15k	biokg
    U   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    F   x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    C+F x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    BST x.xx  x.xx  x.xx  x.xx  x.xx  x.xx    x.xx  x.xx
    """

    dimensions_cfg: str = ConfigCanonicalizer.get_dimensions(
        [str(in_dim), str(out_dim), str(num_heads)], "ax_in.ax_out.ax_head"
    )
    return _calc_best(
        all_time_records_per_dimension_cfg,
        dimensions_cfg,
        all_status_records_per_dimension_cfg,
    )


def calc_worst_mean_best(
    all_HET_time_records_per_dimension_cfg: "dict[str, BenchAllRecords]",
    all_baseline_time_records_per_dimension_cfg: "dict[str, BenchAllRecords]",
    in_dim: int,
    out_dim: int,
    num_heads: int,
    unoptimized_cfg: str = "$UNOPT",
    most_optimized_cfg: str = "$BEST",
):
    """
    An example output:
          Training					Inference
      #degradation	worst	mean	best	#oom by competitors	#degradation	worst	mean	best	#oom by competitors
                              unoptimized
      RGCN	1.00	0.93	1.80	4.59	2.00	1.00	0.97	1.44	3.74	0.00
      RGAT	0.00	4.36	4.93	5.59	6.00	0.00	5.31	6.39	7.76	2.00
      HGT	1.00	1.66	1.88	0.98	2.00	0.00	0.77	1.19	1.98	2.00
                              most optimized
      RGCN	1.00	0.93	1.80	4.59	2.00	1.00	0.97	1.44	3.74	0.00
      RGAT	0.00	4.36	4.93	5.59	6.00	0.00	5.31	6.39	7.76	2.00
      HGT	1.00	1.66	1.88	0.98	2.00	0.00	0.77	1.19	1.98	2.00
    """
    dimension_cfg = ConfigCanonicalizer.get_dimensions(
        [str(in_dim), str(out_dim), str(num_heads)], "ax_in.ax_out.ax_head"
    )
    HET_time_records: BenchAllRecords = all_HET_time_records_per_dimension_cfg[
        dimension_cfg
    ]
    baseline_time_records: BenchAllRecords = (
        all_baseline_time_records_per_dimension_cfg[dimension_cfg]
    )
    result_csv: "list[list[str]]" = [["UNOPTIMIZED"]]
    result_csv += _calc_worst_mean_best(
        HET_time_records, baseline_time_records, unoptimized_cfg
    )
    result_csv += [[]]
    result_csv += [["MOST OPTIMIZED"]]
    result_csv += _calc_worst_mean_best(
        HET_time_records, baseline_time_records, most_optimized_cfg
    )
    return result_csv


def _calc_worst_mean_best(
    HET_time_records: BenchAllRecords,
    baseline_time_records: BenchAllRecords,
    HET_cfg: str,
) -> "list[list[str]]":
    result_csv: "list[list[str]]" = [
        ["Model"] + ["Training"] * 6 + ["Inference"] * 6,
        ["Model"]
        + [
            "#degradation",
            "worst",
            "mean",
            "best",
            "HET #oom",
            "Baseline #oom",
        ]
        * 2,
    ]
    # All model order in tables are reverse alphabetic order
    for model in sorted(HET_time_records.get_all_model(), reverse=True):
        row: list[str] = [model]
        for mode in ["training", "inference"]:
            speed_ups: "list[float]" = []
            # worst_ratio: float = 0.0
            # best_ratio: float = float("inf")
            # mean_ratio: float = 0.0
            num_oom_baseline: int = 0
            num_oom_HET: int = 0
            num_degradation: int = 0
            for dataset in HET_time_records.get_all_dataset():
                best_baseline = baseline_time_records.get_record(
                    mode, dataset, model, "$BEST"
                )
                best_HET = HET_time_records.get_record(
                    mode, dataset, model, HET_cfg
                )
                if not is_float(best_baseline):
                    num_oom_baseline += 1
                if not is_float(best_HET):
                    num_oom_HET += 1
                if is_float(best_baseline) and is_float(best_HET):
                    speed_ups.append(float(best_baseline) / float(best_HET))
                    if speed_ups[-1] < 1.0:
                        num_degradation += 1
            worst: float = (
                min(speed_ups) if len(speed_ups) > 0 else float("inf")
            )
            best: float = max(speed_ups) if len(speed_ups) > 0 else 0.0
            geomean = (
                float(np.array(speed_ups).prod() ** (1.0 / len(speed_ups)))
                if len(speed_ups) > 0
                else 0.0
            )
            row += [
                str(num_degradation),
                str(worst),
                str(geomean),
                str(best),
                str(num_oom_HET),
                str(num_oom_baseline),
            ]
        result_csv.append(row)
    return result_csv


def calc_opt_matrix(
    all_HET_time_records_per_dimension_cfg: "dict[str,BenchAllRecords]",
    in_dim: int,
    out_dim: int,
    num_heads: int,
    unoptimized_cfg: str = "$UNOPT",
) -> "list[list[str]]":
    """
    An example output:
        Training Opt.			Inference Opt.
        C	F	C+F	C	F	C+F
                    RGAT
        aifb	0.84	1.17	0.85	1.04	1.24	1.12
      mutag	0.76	1.14	0.80	1.16	1.25	1.33
      bgs	0.94	1.20	1.06	1.17	1.38	1.41
      am	0.85	1.15	0.93	0.94	1.34	1.08
      mag	1*	OOM	1.02	1*	OOM	1.02
      wikikg2	1*	OOM	1.08	1*	OOM	1.03
      fb15k	1.29	1.25	1.40	1.58	1.40	1.79
      biokg	2.16	1.26	2.20	1.93	1.41	1.97
      AVERAGE	1.06	1.19	1.13	1.26	1.33	1.41
                    HGT
        aifb	1.88	1.26	1.86	1.80	1.82	1.61
      mutag	1.28	1.16	1.32	1.46	1.54	1.36
      bgs	1.20	1.13	1.19	1.27	1.26	1.07
      am	1.11	1.10	1.07	1.29	1.30	0.93
      mag	1.08	1.08	1.14	1.11	1.11	0.90
      wikikg2	1.10	1.09	1.20	1.13	1.13	0.99
      fb15k	1.24	1.14	1.31	1.29	1.29	1.20
      biokg	1.04	1.02	1.15	1.05	1.05	0.95
      AVERAGE	1.22	1.12	1.26	1.28	1.29	1.11
    """
    # TODO
    # 1) Transpose the result of calc_best_HET_and_show_all, 2) calculate the speed up ratio for each cell, and then 3) calculate the average
    dimension_cfg = ConfigCanonicalizer.get_dimensions(
        [str(in_dim), str(out_dim), str(num_heads)], "ax_in.ax_out.ax_head"
    )
    HET_time_records: BenchAllRecords = all_HET_time_records_per_dimension_cfg[
        dimension_cfg
    ]
    return _calc_opt_matrix(HET_time_records, unoptimized_cfg)


def _calc_opt_matrix(
    HET_time_records: BenchAllRecords, unoptimized_cfg: str
) -> "list[list[str]]":
    """
    by default unoptimized_cfg is empty string "" (specified in calc_opt_matrix)
    """
    num_configs = len(
        HET_time_records.get_all_config().difference(
            {unoptimized_cfg, "$BEST", "$BESTCONFIG"}
        )
    )

    result_csv: "list[list[str]]" = [
        ["Model", "Dataset"]
        + ["Training Opt."] * num_configs
        + ["Inference Opt."] * num_configs
    ]

    # All model order in tables are reverse alphabetic order
    for model in sorted(HET_time_records.get_all_model(), reverse=True):
        csv_for_current_model: "list[list[str]]" = [
            ["Model", "Dataset"]
            + sorted(
                [
                    config
                    for config in HET_time_records.get_all_config().difference(
                        {unoptimized_cfg, "$BEST", "$BESTCONFIG"}
                    )
                ]
            )
            * 2
        ]
        for dataset in sorted(HET_time_records.get_all_dataset()):
            # Each row shows runs on one dataset with different configs
            row: list[str] = [model, dataset]
            for mode in ["training", "inference"]:
                unoptimized = HET_time_records.get_record(
                    mode, dataset, model, unoptimized_cfg
                )
                if not is_float(unoptimized):
                    # find the largest time
                    for config in HET_time_records.get_all_config().difference(
                        {unoptimized_cfg, "$BEST", "$BESTCONFIG"}
                    ):
                        curr = HET_time_records.get_record(
                            mode, dataset, model, config
                        )
                        if is_float(curr):
                            if is_float(unoptimized):
                                unoptimized = max(unoptimized, curr)
                            else:
                                unoptimized = curr
                if not is_float(unoptimized):
                    # all OOM
                    for config in HET_time_records.get_all_config().difference(
                        {unoptimized_cfg, "$BEST", "$BESTCONFIG"}
                    ):
                        row.append("OOM")
                    continue
                unoptimized = float(unoptimized)
                for config in sorted(
                    HET_time_records.get_all_config().difference(
                        {unoptimized_cfg, "$BEST", "$BESTCONFIG"}
                    )
                ):
                    curr = HET_time_records.get_record(
                        mode, dataset, model, config
                    )
                    if not is_float(curr):
                        row.append("OOM")
                    else:
                        curr = float(curr)
                        row.append(str(unoptimized / curr))
            csv_for_current_model.append(row)
        average_row = [model, "AVERAGE"]
        # Calculate the average
        for column_idx in range(
            2, len(csv_for_current_model[-1])
        ):  # not involving model and dataset
            numerics = [
                float(row[column_idx])
                for row in csv_for_current_model
                if column_idx < len(row) and is_float(row[column_idx])
            ]
            geomean = (
                float(np.array(numerics).prod() ** (1.0 / len(numerics)))
                if len(numerics) > 0
                else 0.0
            )
            average_row.append(str(geomean))

        csv_for_current_model.append(average_row)
        csv_for_current_model.append([])
        result_csv += csv_for_current_model
    return result_csv


if __name__ == "__main__":
    # TODO: filter out "Fusion.Compact" and ".Compact"

    assert is_pwd_het_dev_root(), "Please run this script at het_dev root"
    # Load data from the results folder
    graphiler_results_dir = ask_subdirectory(
        "misc/artifacts", "graphiler_", RESULTS_DIR
    )
    print("Obtaining results from", graphiler_results_dir)
    graphiler_names_and_info = (
        extract_graphiler_and_its_baselines_results_from_folder(
            graphiler_results_dir
        )
    )
    het_results_dir = ask_subdirectory(
        "misc/artifacts", "benchmark_all_", RESULTS_DIR
    )
    print("Obtaining results from", het_results_dir)
    het_names_and_info = extract_het_results_from_folder(het_results_dir)
    all_baseline_records = BenchAllRecords.load_baseline_results_from_uploader(
        graphiler_names_and_info
    )
    (
        all_het_time_records,
        all_het_dram_records,
        all_het_status_records,
    ) = BenchAllRecords.load_HET_results_from_uploader(
        het_names_and_info, "flag_mul.flag_compact.ax_in.ax_out.ax_head"
    )

    # Draw tables
    tab_best_baseline = calc_best_baselines_and_show_all(
        all_baseline_records, 64, 64, 1
    )
    tab_best_het = calc_best_HET_and_show_all(
        all_het_time_records, all_het_status_records, 64, 64, 1
    )
    tab_worst_mean_best = calc_worst_mean_best(
        all_het_time_records, all_baseline_records, 64, 64, 1
    )
    tab_opt_matrix = calc_opt_matrix(all_het_time_records, 64, 64, 1)

    # upload it
    worksheet_title = f"[{get_pretty_hostname()}]tables.{graphiler_results_dir.split('/')[-1]}.{het_results_dir.split('/')[-1]}"
    try:
        worksheet = create_worksheet(SPREADSHEET_URL, worksheet_title)

        # Increase the size of grid if needed
        default_rows = 100
        default_cols = 26
        total_num_rows = max(
            [
                count_rows(tab_best_baseline),
                count_rows(tab_best_het),
                count_rows(tab_worst_mean_best),
                count_rows(tab_opt_matrix),
            ]
        )
        total_num_cols = (
            count_cols(tab_best_baseline)
            + count_cols(tab_best_het)
            + count_cols(tab_worst_mean_best)
            + count_cols(tab_opt_matrix)
        )
        if total_num_rows > default_rows or total_num_cols > default_cols:
            worksheet.resize(total_num_rows, total_num_cols)
        update_gspread(
            tab_best_baseline,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_baseline),
                count_cols(tab_best_baseline),
                0,
                0,
            ),
        )
        update_gspread(
            tab_best_het,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_best_het),
                count_cols(tab_best_het),
                0,
                count_cols(tab_best_baseline),
            ),
        )
        update_gspread(
            tab_worst_mean_best,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_worst_mean_best),
                count_cols(tab_worst_mean_best),
                0,
                count_cols(tab_best_baseline) + count_cols(tab_best_het),
            ),
        )
        update_gspread(
            tab_opt_matrix,
            worksheet,
            cell_range=get_cell_range_from_A1(
                count_rows(tab_opt_matrix),
                count_cols(tab_opt_matrix),
                0,
                count_cols(tab_best_baseline)
                + count_cols(tab_best_het)
                + count_cols(tab_worst_mean_best),
            ),
        )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
