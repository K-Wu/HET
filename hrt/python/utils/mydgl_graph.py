#!/usr/bin/env python3

# NB: this class stores type in list and assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that.
from __future__ import annotations
from ..kernels import K

from .. import utils
from .. import utils_lite
import functools

#    convert_integrated_csr_to_separate_csr,
#    convert_integrated_csr_to_separate_coo,
#    convert_integrated_coo_to_separate_csr,
#    convert_integrated_coo_to_separate_coo,
#    transpose_csr

import torch
import torch.jit

import dgl

# This maps ScriptedMyDGLGraph data member to MyDGLGraph function that produces it.
scripted_member_to_function: dict[str, str] = {
    "transposed_sparse_format": "transpose",
    "transposed_coo": "transpose",
    "separate_unique_node_indices": (
        "generate_separate_unique_node_indices_for_each_etype"
    ),
    # TODO: add separate CSR
    "separate_unique_node_indices_single_sided": (
        "generate_separate_unique_node_indices_single_sided_for_each_etype"
    ),
    "separate_unique_node_indices_inverse_idx": (
        "generate_separate_unique_node_indices_for_each_etype"
    ),
    "separate_unique_node_indices_single_sided_inverse_idx": (
        "generate_separate_unique_node_indices_single_sided_for_each_etype"
    ),
    "separate_coo_original": "generate_separate_coo_adj_for_each_etype",
    "separate_csr_original": "generate_separate_csr_adj_for_each_etype",
}

# Store this in list to make sure the order is preserved.
graph_data_key_to_function: list[tuple[tuple[str, ...], str]] = [
    (("transposed",), "transpose"),
    (
        ("separate", "coo"),
        "generate_separate_coo_adj_for_each_etype_both_transposed_and_non_transposed",
    ),
    (("separate", "csr"), "generate_separate_csr_adj_for_each_etype"),
    (("separate", "unique_node_indices"), "separate_unique_node_indices"),
    (
        (
            "separate",
            "unique_node_indices_single_sided",
        ),
        "generate_separate_unique_node_indices_single_sided_for_each_etype",
    ),
]


def recursive_apply_to_each_tensor_in_dict(func, dict_var, filter_key_set):
    for second_key in dict_var:
        if second_key in filter_key_set:
            continue
        if type(dict_var[second_key]) == torch.Tensor:
            dict_var[second_key] = func(dict_var[second_key])
        elif type(dict_var[second_key]) == dict:
            recursive_apply_to_each_tensor_in_dict(
                func, dict_var[second_key], filter_key_set
            )
        else:
            print(
                "WARNING in apply_to_each_tensor_in_dict: second_key"
                " {second_key} unknown type {type}".format(
                    second_key=second_key, type=type(dict_var[second_key])
                )
            )


class MyDGLGraph:
    def __init__(self):
        self.graph_data = dict()
        self.sequential_eids_format = ""

    @torch.no_grad()
    def _get_num_nodes(self):
        assert "original" in self.graph_data
        if "node_type_offsets" in self.graph_data["original"]:
            return int(self.graph_data["original"]["node_type_offsets"][-1])
        if "row_ptrs" in self.graph_data["original"]:
            return int(
                max(
                    self.graph_data["original"]["row_ptrs"].numel() - 1,
                    int(self.graph_data["original"]["col_indices"].max()) + 1,
                )
            )
        else:
            assert (
                "row_indices" in self.graph_data["original"]
            ), "row_idx not exists"
            assert (
                "col_indices" in self.graph_data["original"]
            ), "col_idx not exists"
            return int(
                max(
                    int(self.graph_data["original"]["row_indices"].max()) + 1,
                    int(self.graph_data["original"]["col_indices"].max()) + 1,
                )
            )

    def get_num_nodes(self):
        if getattr(self, "num_nodes", None) is None:
            self.num_nodes = self._get_num_nodes()
        return self.num_nodes

    def get_num_ntypes(self):
        assert "original" in self.graph_data
        if "node_type_offsets" in self.graph_data["original"]:
            return len(self.graph_data["original"]["node_type_offsets"]) - 1
        else:
            return 1

    @torch.no_grad()
    def _get_num_rels(self):
        result = int(self.graph_data["original"]["rel_types"].max().item()) + 1
        if result <= 1:
            print("WARNING: get_num_rels <= 1")
        if (
            "legacy_metadata_from_dgl" in self.graph_data
            and "canonical_etypes"
            in self.graph_data["legacy_metadata_from_dgl"]
        ):
            if (
                len(
                    self.graph_data["legacy_metadata_from_dgl"][
                        "canonical_etypes"
                    ]
                )
                <= 1
            ):
                print("WARNING: len(canonical_etypes) <= 1")
                print(
                    self.graph_data["legacy_metadata_from_dgl"][
                        "canonical_etypes"
                    ]
                )
            if (
                len(
                    self.graph_data["legacy_metadata_from_dgl"][
                        "canonical_etypes"
                    ]
                )
                != result
            ):
                print(
                    "WARNING: len(canonical_etypes) != get_num_rels",
                    len(
                        self.graph_data["legacy_metadata_from_dgl"][
                            "canonical_etypes"
                        ]
                    ),
                    result,
                )
        return result

    def get_num_rels(self):
        if getattr(self, "num_rels", None) is None:
            self.num_rels = self._get_num_rels()
        return self.num_rels

    def _get_num_edges(self):
        return self.graph_data["original"]["rel_types"].numel()

    def get_num_edges(self):
        if getattr(self, "num_edges", None) is None:
            self.num_edges = self._get_num_edges()
        return self.num_edges

    def apply_to_each_tensor(self, func):
        recursive_apply_to_each_tensor_in_dict(
            func, self.graph_data, ["legacy_metadata_from_dgl"]
        )

    def to_(self, device):
        """The _ in the end suggests that this function is in-place, i.e., the change apply to the original object."""
        self.apply_to_each_tensor(lambda x: x.to(device))
        return self

    def cuda_(self):
        """The _ in the end suggests that this function is in-place, i.e., the change apply to the original object."""
        self.apply_to_each_tensor(lambda x: x.cuda())
        return self

    def cpu_(self):
        """The _ in the end suggests that this function is in-place, i.e., the change apply to the original object."""
        self.apply_to_each_tensor(lambda x: x.cpu())
        return self

    def contiguous_(self):
        """The _ in the end suggests that this function is in-place, i.e., the change apply to the original object."""
        self.apply_to_each_tensor(lambda x: x.contiguous())
        return self

    def __setitem__(self, key, value):
        self.graph_data[key] = value

    def __getitem__(self, key):
        return self.graph_data[key]

    def __contains__(self, key):
        return key in self.graph_data

    def save_to_disk(self, filename):
        torch.save(self.graph_data, filename)

    def load_from_disk(self, filename):
        self.graph_data = torch.load(filename)

    def get_sparse_format(self, transpose_flag: bool = False):
        original_or_transpose = "transposed" if transpose_flag else "original"
        if "row_ptrs" in self.graph_data[original_or_transpose]:
            return "csr"
        elif "row_indices" in self.graph_data[original_or_transpose]:
            assert (
                "col_indices" in self.graph_data[original_or_transpose]
            ), "col_idx not exists"
            return "coo"
        else:
            raise ValueError("unknown sparse format")

    @torch.no_grad()
    def transpose(self):
        if self.get_sparse_format() == "csr":
            (
                self.graph_data["original"]["row_ptrs"],
                self.graph_data["original"]["col_indices"],
            ) = (
                self.graph_data["original"]["col_indices"],
                self.graph_data["original"]["row_ptrs"],
            )
        elif self.get_sparse_format() == "coo":
            assert (
                "transposed" not in self.graph_data
            ), "transposed already exists"
            self.graph_data["transposed"] = dict()
            self.graph_data["transposed"]["row_indices"] = self.graph_data[
                "original"
            ]["col_indices"]
            self.graph_data["transposed"]["col_indices"] = self.graph_data[
                "original"
            ]["row_indices"]
            self.graph_data["transposed"]["eids"] = self.graph_data[
                "original"
            ]["eids"]
            self.graph_data["transposed"]["rel_types"] = self.graph_data[
                "original"
            ]["rel_types"]
        else:
            assert (
                "transposed" not in self.graph_data
            ), "transposed already exists"
            self.graph_data["transposed"] = dict()
            (
                transposed_rowptr,
                transposed_col_idx,
                transposed_rel_types,
                transposed_eids,
            ) = K.transpose_csr(
                self.graph_data["original"]["row_ptrs"],
                self.graph_data["original"]["col_indices"],
                self.graph_data["original"]["rel_types"],
                self.graph_data["original"]["eids"],
            )
            self.graph_data["transposed"]["row_ptrs"] = transposed_rowptr
            self.graph_data["transposed"]["col_indices"] = transposed_col_idx
            self.graph_data["transposed"]["rel_types"] = transposed_rel_types
            self.graph_data["transposed"]["eids"] = transposed_eids
        return self

    def get_original_coo(self) -> dict[str, torch.Tensor]:
        original_coo = dict()
        original_coo["rel_types"] = self.graph_data["original"]["rel_types"]
        original_coo["row_indices"] = self.graph_data["original"][
            "row_indices"
        ]
        original_coo["col_indices"] = self.graph_data["original"][
            "col_indices"
        ]
        original_coo["eids"] = self.graph_data["original"]["eids"]
        return original_coo

    def get_out_csr(self) -> dict[str, torch.Tensor]:
        out_csr = dict()
        out_csr["rel_types"] = self.graph_data["original"]["rel_types"]
        out_csr["row_ptrs"] = self.graph_data["original"]["row_ptrs"]
        out_csr["col_indices"] = self.graph_data["original"]["col_indices"]
        out_csr["eids"] = self.graph_data["original"]["eids"]
        return out_csr

    def get_transposed_coo(self) -> dict[str, torch.Tensor]:
        transposed_coo = dict()
        transposed_coo["rel_types"] = self.graph_data["transposed"][
            "rel_types"
        ]
        transposed_coo["row_indices"] = self.graph_data["transposed"][
            "row_indices"
        ]
        transposed_coo["col_indices"] = self.graph_data["transposed"][
            "col_indices"
        ]
        transposed_coo["eids"] = self.graph_data["transposed"]["eids"]
        return transposed_coo

    def get_in_csr(self) -> dict[str, torch.Tensor]:
        in_csr = dict()
        in_csr["rel_types"] = self.graph_data["transposed"]["rel_types"]
        in_csr["row_ptrs"] = self.graph_data["transposed"]["row_ptrs"]
        in_csr["col_indices"] = self.graph_data["transposed"]["col_indices"]
        in_csr["eids"] = self.graph_data["transposed"]["eids"]
        return in_csr

    def get_original_node_type_offsets(self) -> torch.Tensor:
        return self.graph_data["original"]["node_type_offsets"]

    def get_separate_coo_original(self) -> dict[str, torch.Tensor]:
        separate_coo_original = dict()
        separate_coo_original["rel_ptrs"] = self.graph_data["separate"]["coo"][
            "original"
        ]["rel_ptrs"]
        separate_coo_original["row_indices"] = self.graph_data["separate"][
            "coo"
        ]["original"]["row_indices"]
        separate_coo_original["col_indices"] = self.graph_data["separate"][
            "coo"
        ]["original"]["col_indices"]
        separate_coo_original["eids"] = self.graph_data["separate"]["coo"][
            "original"
        ]["eids"]
        return separate_coo_original

    def get_separate_csr_original(self) -> dict[str, torch.Tensor]:
        separate_csr_original = dict()
        separate_csr_original["rel_ptrs"] = self.graph_data["separate"]["csr"][
            "original"
        ]["rel_ptrs"]
        separate_csr_original["row_ptrs"] = self.graph_data["separate"]["csr"][
            "original"
        ]["row_ptrs"]
        separate_csr_original["col_indices"] = self.graph_data["separate"][
            "csr"
        ]["original"]["col_indices"]
        separate_csr_original["eids"] = self.graph_data["separate"]["csr"][
            "original"
        ]["eids"]
        return separate_csr_original

    def get_separate_unique_node_indices(self) -> dict[str, torch.Tensor]:
        separate_unique_node_indices = dict()
        separate_unique_node_indices["rel_ptrs"] = self.graph_data["separate"][
            "unique_node_indices"
        ]["rel_ptrs"]
        separate_unique_node_indices["node_indices"] = self.graph_data[
            "separate"
        ]["unique_node_indices"]["node_indices"]
        return separate_unique_node_indices

    def get_separate_unique_node_indices_single_sided(
        self,
    ) -> dict[str, torch.Tensor]:
        separate_unique_node_indices_single_sided = dict()
        separate_unique_node_indices_single_sided[
            "node_indices_row"
        ] = self.graph_data["separate"]["unique_node_indices_single_sided"][
            "node_indices_row"
        ]
        separate_unique_node_indices_single_sided[
            "rel_ptrs_row"
        ] = self.graph_data["separate"]["unique_node_indices_single_sided"][
            "rel_ptrs_row"
        ]
        separate_unique_node_indices_single_sided[
            "node_indices_col"
        ] = self.graph_data["separate"]["unique_node_indices_single_sided"][
            "node_indices_col"
        ]
        separate_unique_node_indices_single_sided[
            "rel_ptrs_col"
        ] = self.graph_data["separate"]["unique_node_indices_single_sided"][
            "rel_ptrs_col"
        ]
        return separate_unique_node_indices_single_sided

    def get_separate_unique_node_indices_inverse_idx(
        self,
    ) -> dict[str, torch.Tensor]:
        ret = dict()
        ret["rel_ptrs"] = self.graph_data["separate"]["unique_node_indices"][
            "rel_ptrs"
        ]
        ret["inverse_indices"] = self.graph_data["separate"][
            "unique_node_indices"
        ]["inverse_indices"]
        return ret

    def get_separate_unique_node_indices_single_sided_inverse_idx(
        self,
    ) -> dict[str, torch.Tensor]:
        ret = dict()
        ret["rel_ptrs_row"] = self.graph_data["separate"][
            "unique_node_indices_single_sided"
        ]["rel_ptrs_row"]
        ret["inverse_indices_row"] = self.graph_data["separate"][
            "unique_node_indices_single_sided"
        ]["inverse_indices_row"]
        ret["inverse_indices_col"] = self.graph_data["separate"][
            "unique_node_indices_single_sided"
        ]["inverse_indices_col"]
        return ret

    def to_script_object(self) -> utils.ScriptedMyDGLGraph:
        num_nodes = self.get_num_nodes()
        num_ntypes = self.get_num_ntypes()
        num_rels = self.get_num_rels()
        num_edges = self.get_num_edges()
        sparse_format = self.get_sparse_format()
        if "transposed" in self.graph_data:
            transposed_sparse_format = self.get_sparse_format(True)
        else:
            transposed_sparse_format = None

        if sparse_format == "coo":
            original_coo = self.get_original_coo()
            out_csr = None
        elif sparse_format == "csr":
            out_csr = self.get_out_csr()
            original_coo = None
        else:
            raise ValueError("unknown sparse format")

        if transposed_sparse_format == "coo":
            transposed_coo = self.get_transposed_coo()
            in_csr = None
        elif transposed_sparse_format == "csr":
            in_csr = self.get_in_csr()
            transposed_coo = None
        else:
            in_csr = None
            transposed_coo = None

        if "node_type_offsets" in self.graph_data["original"]:
            original_node_type_offsets = self.get_original_node_type_offsets()
        else:
            original_node_type_offsets = None

        separate_coo_original = None
        separate_csr_original = None
        separate_unique_node_indices = None
        separate_unique_node_indices_single_sided = None
        separate_unique_node_indices_inverse_idx = None
        separate_unique_node_indices_single_sided_inverse_idx = None
        if "separate" in self.graph_data:
            if (
                "coo" in self.graph_data["separate"]
                and "original" in self.graph_data["separate"]["coo"]
            ):
                separate_coo_original = self.get_separate_coo_original()
            if (
                "csr" in self.graph_data["separate"]
                and "original" in self.graph_data["separate"]["csr"]
            ):
                separate_csr_original = self.get_separate_csr_original()
            if "unique_node_indices" in self.graph_data["separate"]:
                separate_unique_node_indices = (
                    self.get_separate_unique_node_indices()
                )
                if (
                    "inverse_indices"
                    in self.graph_data["separate"]["unique_node_indices"]
                ):
                    separate_unique_node_indices_inverse_idx = (
                        self.get_separate_unique_node_indices_inverse_idx()
                    )
            if (
                "unique_node_indices_single_sided"
                in self.graph_data["separate"]
            ):
                separate_unique_node_indices_single_sided = (
                    self.get_separate_unique_node_indices_single_sided()
                )
                if (
                    "inverse_indices_row"
                    in self.graph_data["separate"][
                        "unique_node_indices_single_sided"
                    ]
                ):
                    separate_unique_node_indices_single_sided_inverse_idx = (
                        self.get_separate_unique_node_indices_single_sided_inverse_idx()
                    )

        return utils.ScriptedMyDGLGraph(
            num_nodes,
            num_ntypes,
            num_rels,
            num_edges,
            sparse_format,
            transposed_sparse_format,
            original_coo,
            transposed_coo,
            in_csr,
            out_csr,
            original_node_type_offsets,
            separate_unique_node_indices,
            separate_unique_node_indices_single_sided,
            separate_unique_node_indices_inverse_idx,
            separate_unique_node_indices_single_sided_inverse_idx,
            separate_coo_original,
            separate_csr_original,
        )

    @torch.no_grad()
    def generate_separate_csr_adj_for_each_etype(
        self, transposed_flag, sorted_rel_eid_flag=True
    ):
        # store rel_ptr, row_ptr, col_idx, eids in self.graph_data["separate"]["csr"]["original"] and self.graph_data["separate"]["csr"]["transposed"]
        if sorted_rel_eid_flag:
            assert 0, (
                "WARNING: sorted_rel_eid_flag is True, which is not supported"
                " for separate_csr_adj generation."
            )
        # first make sure graph data, as torch tensors, are on cpu
        self.cpu_()

        if "separate" not in self.graph_data:
            self.graph_data["separate"] = dict()
        else:
            print(
                "WARNING : in generating_separate separate already exists,"
                " will be overwritten"
            )
        if "csr" not in self.graph_data["separate"]:
            self.graph_data["separate"]["csr"] = dict()
        else:
            print(
                "WARNING : in generating_separate csr already exists, will be"
                " overwritten"
            )

        # make sure the corresponding key in graph data is assigned an empty container
        if transposed_flag:
            if "transposed" not in self.graph_data["separate"]["csr"]:
                self.graph_data["separate"]["csr"]["transposed"] = dict()
            else:
                print(
                    "WARNING: in generating_separate transposed already"
                    " exists, will be overwritten"
                )
            if "transposed" not in self.graph_data:
                self.transpose()
            assert self.get_sparse_format(
                transpose_flag=False
            ) == self.get_sparse_format(transpose_flag=True)

        else:
            if "original" not in self.graph_data["separate"]["csr"]:
                self.graph_data["separate"]["csr"]["original"] = dict()
            else:
                print(
                    "WARNING : in generating_separate original already exists,"
                    " will be overwritten"
                )

        original_or_transposed = (
            "transposed" if transposed_flag else "original"
        )

        # then call C++ wrapper function to do the job
        if self.get_sparse_format() == "csr":
            (
                separate_csr_rel_ptrs,
                separate_csr_row_ptrs,
                separate_csr_col_idx,
                separate_csr_eids,
            ) = K.convert_integrated_csr_to_separate_csr(
                self.graph_data[original_or_transposed]["row_ptrs"],
                self.graph_data[original_or_transposed]["col_indices"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        elif self.get_sparse_format() == "coo":
            (
                separate_csr_rel_ptrs,
                separate_csr_row_ptrs,
                separate_csr_col_idx,
                separate_csr_eids,
            ) = K.convert_integrated_coo_to_separate_csr(
                self.graph_data[original_or_transposed]["row_indices"],
                self.graph_data[original_or_transposed]["col_indices"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        else:
            raise ValueError("unknown sparse format")
        self.graph_data["separate"]["csr"][original_or_transposed][
            "rel_ptrs"
        ] = separate_csr_rel_ptrs
        self.graph_data["separate"]["csr"][original_or_transposed][
            "row_ptrs"
        ] = separate_csr_row_ptrs
        self.graph_data["separate"]["csr"][original_or_transposed][
            "col_indices"
        ] = separate_csr_col_idx
        self.graph_data["separate"]["csr"][original_or_transposed][
            "eids"
        ] = separate_csr_eids

    @torch.no_grad()
    def generate_separate_coo_adj_for_each_etype_both_transposed_and_non_transposed(
        self,
    ):
        """This is defined for the ease of get_funcs_to_propagate_and_produce_metadata"""
        # TODO: refine the implementation for get_funcs_to_propagate_and_produce_metadata
        self.generate_separate_coo_adj_for_each_etype(
            transposed_flag=False, rel_eid_sorted_flag=True
        )
        self.generate_separate_coo_adj_for_each_etype(
            transposed_flag=True, rel_eid_sorted_flag=True
        )

    @torch.no_grad()
    def generate_separate_coo_adj_for_each_etype(
        self, transposed_flag, rel_eid_sorted_flag=True
    ):
        # store rel_ptr, row_idx, col_idx, eids in self.graph_data["separate"]["coo"]["original"] and self.graph_data["separate"]["coo"]["transposed"]
        # first make sure graph data, as torch tensors, are on cpu
        self.cpu_()

        if "separate" not in self.graph_data:
            self.graph_data["separate"] = dict()
        else:
            print(
                "WARNING : in generating_separate separate already exists, may"
                " be overwritten"
            )
        if "coo" not in self.graph_data["separate"]:
            self.graph_data["separate"]["coo"] = dict()
        else:
            print(
                "WARNING : in generating_separate coo already exists, may be"
                " overwritten"
            )

        if transposed_flag:
            if "transposed" not in self.graph_data["separate"]["coo"]:
                self.graph_data["separate"]["coo"]["transposed"] = dict()
            else:
                print(
                    "WARNING: in generating_separate transposed already"
                    " exists, may be overwritten"
                )
            if "transposed" not in self.graph_data:
                self.transpose()
            assert self.get_sparse_format(
                transpose_flag=False
            ) == self.get_sparse_format(transpose_flag=True)

        else:
            if "original" not in self.graph_data["separate"]["coo"]:
                self.graph_data["separate"]["coo"]["original"] = dict()
            else:
                print(
                    "WARNING : in generating_separate original already exists,"
                    " may be overwritten"
                )

        original_or_transposed = (
            "transposed" if transposed_flag else "original"
        )

        if self.get_sparse_format() == "csr":
            (
                separate_coo_rel_ptrs,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = K.convert_integrated_csr_to_separate_coo(
                self.graph_data[original_or_transposed]["row_ptrs"],
                self.graph_data[original_or_transposed]["col_indices"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        elif self.get_sparse_format() == "coo":
            (
                separate_coo_rel_ptrs,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = K.convert_integrated_coo_to_separate_coo(
                self.graph_data[original_or_transposed]["row_indices"],
                self.graph_data[original_or_transposed]["col_indices"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
                self.get_num_nodes(),
                self.get_num_rels(),
            )
        else:
            raise ValueError("unknown sparse format")

        if rel_eid_sorted_flag:
            # sort the coo by rel_type and eid
            (
                separate_coo_rel_ptrs,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = utils.sort_coo_by_etype_eids_torch_tensors(
                separate_coo_rel_ptrs,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            )
        self.graph_data["separate"]["coo"][original_or_transposed][
            "rel_ptrs"
        ] = separate_coo_rel_ptrs
        self.graph_data["separate"]["coo"][original_or_transposed][
            "row_indices"
        ] = separate_coo_row_idx
        self.graph_data["separate"]["coo"][original_or_transposed][
            "col_indices"
        ] = separate_coo_col_idx
        self.graph_data["separate"]["coo"][original_or_transposed][
            "eids"
        ] = separate_coo_eids

    @torch.no_grad()
    def get_canonicalize_eid_mapping(
        self, target_sequential_eids_format="separate_coo"
    ):
        @torch.no_grad()
        def _get_old_to_sequential_mapping(eid_tensor):
            return dict(zip(eid_tensor.tolist(), range(eid_tensor.shape[0])))

        old_to_new_eid_mapping = dict()
        if target_sequential_eids_format == "separate_coo":
            old_to_new_eid_mapping = _get_old_to_sequential_mapping(
                self.graph_data["separate"]["coo"]["original"]["eids"]
            )
        elif target_sequential_eids_format == "separate_csr":
            old_to_new_eid_mapping = _get_old_to_sequential_mapping(
                self.graph_data["separate"]["csr"]["original"]["eids"]
            )
        elif target_sequential_eids_format == "integrated_coo":
            old_to_new_eid_mapping = _get_old_to_sequential_mapping(
                self.graph_data["original"]["eids"].tolist()
            )
        elif target_sequential_eids_format == "integrated_csr":
            old_to_new_eid_mapping = _get_old_to_sequential_mapping(
                self.graph_data["original"]["eids"].tolist()
            )
        else:
            raise ValueError("unknown sequential eids format")
        return old_to_new_eid_mapping

    @torch.no_grad()
    def canonicalize_eids(self, target_sequential_eids_format="separate_coo"):
        @torch.no_grad()
        def _canonicalize_eids(old_to_new_eid_mapping, dict_prefix):
            dict_prefix["eids"] = torch.tensor(
                list(
                    map(
                        old_to_new_eid_mapping.get,
                        dict_prefix["eids"].tolist(),
                    )
                )
            )

        if target_sequential_eids_format == self.sequential_eids_format:
            print("WARNING: already in target sequential eids format")
            return

        # step 1 get the mapping
        old_to_new_eid_mapping = self.get_canonicalize_eid_mapping(
            target_sequential_eids_format
        )

        # step 2 apply the mapping
        if (
            "separate" in self.graph_data
            and "coo" in self.graph_data["separate"]
        ):
            if "original" in self.graph_data["separate"]["coo"]:
                _canonicalize_eids(
                    old_to_new_eid_mapping,
                    self.graph_data["separate"]["coo"]["original"],
                )
            if "transposed" in self.graph_data["separate"]["coo"]:
                _canonicalize_eids(
                    old_to_new_eid_mapping,
                    self.graph_data["separate"]["coo"]["transposed"],
                )
        if (
            "separate" in self.graph_data
            and "csr" in self.graph_data["separate"]
        ):
            if "original" in self.graph_data["separate"]["csr"]:
                _canonicalize_eids(
                    old_to_new_eid_mapping,
                    self.graph_data["separate"]["csr"]["original"],
                )
            if "transposed" in self.graph_data["separate"]["csr"]:
                _canonicalize_eids(
                    old_to_new_eid_mapping,
                    self.graph_data["separate"]["csr"]["transposed"],
                )
        if "original" in self.graph_data:
            _canonicalize_eids(
                old_to_new_eid_mapping, self.graph_data["original"]
            )
        if "transposed" in self.graph_data:
            _canonicalize_eids(
                old_to_new_eid_mapping, self.graph_data["transposed"]
            )
        self.sequential_eids_format = target_sequential_eids_format

    @torch.no_grad()
    def generate_separate_unique_node_indices_single_sided_for_each_etype(
        self, produce_inverse_idx=True
    ):
        if (
            "separate" not in self.graph_data
            or "coo" not in self.graph_data["separate"]
        ):
            raise ValueError(
                "separate coo graph data not found, please generate it first"
            )
        if produce_inverse_idx:
            # canonicalizing all formats' eid by setting separate coo's as sequential by a[b.long()].long()
            # In this way, we don't need extra step to canonicalize the inverse idx
            self.canonicalize_eids(
                target_sequential_eids_format="separate_coo"
            )

        (
            result_node_indices_row,
            result_rel_ptrs_row,
            result_node_indices_col,
            result_rel_ptrs_col,
            result_node_indices_row_reverse_idx,
            result_node_indices_col_reverse_idx,
        ) = utils_lite.generate_separate_unique_node_indices_single_sided_for_each_etype(
            self.get_num_rels(),
            self.graph_data["separate"]["coo"]["original"]["rel_ptrs"],
            self.graph_data["separate"]["coo"]["original"]["row_indices"],
            self.graph_data["separate"]["coo"]["original"]["col_indices"],
            get_inverse_idx=produce_inverse_idx,
        )

        if "unique_node_indices_single_sided" in self.graph_data["separate"]:
            print(
                "WARNING: unique_node_indices_single_sided already exists,"
                " will be overwritten"
            )
        self.graph_data["separate"][
            "unique_node_indices_single_sided"
        ] = dict()
        self.graph_data["separate"]["unique_node_indices_single_sided"][
            "node_indices_row"
        ] = result_node_indices_row
        self.graph_data["separate"]["unique_node_indices_single_sided"][
            "rel_ptrs_row"
        ] = result_rel_ptrs_row
        self.graph_data["separate"]["unique_node_indices_single_sided"][
            "node_indices_col"
        ] = result_node_indices_col
        self.graph_data["separate"]["unique_node_indices_single_sided"][
            "rel_ptrs_col"
        ] = result_rel_ptrs_col
        if produce_inverse_idx:
            self.graph_data["separate"]["unique_node_indices_single_sided"][
                "inverse_indices_row"
            ] = result_node_indices_row_reverse_idx
            self.graph_data["separate"]["unique_node_indices_single_sided"][
                "inverse_indices_col"
            ] = result_node_indices_col_reverse_idx

    @torch.no_grad()
    def generate_separate_unique_node_indices_for_each_etype(
        self, produce_inverse_idx=True
    ):
        if (
            "separate" not in self.graph_data
            or "coo" not in self.graph_data["separate"]
        ):
            raise ValueError(
                "separate coo graph data not found, please generate it first"
            )
        if produce_inverse_idx:
            # canonicalizing all formats' eid by setting separate coo's as sequential by a[b.long()].long()
            # In this way, we don't need extra step to canonicalize the inverse idx
            self.canonicalize_eids(
                target_sequential_eids_format="separate_coo"
            )

        (
            result_node_idx,
            result_rel_ptrs,
            result_node_indices_inverse_idx,
        ) = utils_lite.generate_separate_unique_node_indices_for_each_etype(
            self.get_num_rels(),
            self.graph_data["separate"]["coo"]["original"]["rel_ptrs"],
            self.graph_data["separate"]["coo"]["original"]["row_indices"],
            self.graph_data["separate"]["coo"]["original"]["col_indices"],
            get_inverse_idx=produce_inverse_idx,
        )

        if "unique_node_indices" in self.graph_data["separate"]:
            print(
                "WARNING: unique_node_idx already exists, will be overwritten"
            )
        self.graph_data["separate"]["unique_node_indices"] = dict()
        self.graph_data["separate"]["unique_node_indices"][
            "node_indices"
        ] = result_node_idx
        self.graph_data["separate"]["unique_node_indices"][
            "rel_ptrs"
        ] = result_rel_ptrs
        if produce_inverse_idx:
            self.graph_data["separate"]["unique_node_indices"][
                "inverse_indices"
            ] = result_node_indices_inverse_idx
            # TODO: this is only mapping to the separate coo idx, consider canonicalizing separate coo's eid as sequential eid by a[b.long()].long(), in order to keep the current kernel code unchanged

    def import_metadata_from_dgl_heterograph(self, dglgraph):
        assert (
            "legacy_metadata_from_dgl" not in self.graph_data
        ), "legacy_metadata_from_dgl already exists"
        self["legacy_metadata_from_dgl"] = dict()
        self["legacy_metadata_from_dgl"][
            "canonical_etypes"
        ] = dglgraph.canonical_etypes
        self["legacy_metadata_from_dgl"]["ntypes"] = dglgraph.ntypes

        self["legacy_metadata_from_dgl"]["number_of_nodes_per_type"] = dict(
            [
                (ntype, dglgraph.number_of_nodes(ntype))
                for ntype in dglgraph.ntypes
            ]
        )
        # Number of nodes and number of edges in legacy_metadata_from_dgl are unused and causes confusion when propagating metadata to subgraphs
        # self["legacy_metadata_from_dgl"][
        #     "number_of_nodes"
        # ] = dglgraph.number_of_nodes()
        # self["legacy_metadata_from_dgl"][
        #     "number_of_edges"
        # ] = dglgraph.number_of_edges()
        self["legacy_metadata_from_dgl"]["number_of_edges_per_type"] = dict(
            [
                (etype, dglgraph.number_of_edges(etype))
                for etype in dglgraph.canonical_etypes
            ]
        )
        self["legacy_metadata_from_dgl"]["node_dict"] = dict(
            zip(dglgraph.ntypes, range(len(dglgraph.ntypes)))
        )
        self["legacy_metadata_from_dgl"]["edge_dict"] = dict(
            zip(
                dglgraph.canonical_etypes,
                range(len(dglgraph.canonical_etypes)),
            )
        )

    def get_dgl_graph(self, transposed_flag: bool = False):
        if (
            "separate" not in self.graph_data
            or "coo" not in self.graph_data["separate"]
        ):
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )
        if (
            transposed_flag
            and "transposed" not in self.graph_data["separate"]["coo"]
        ):
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )
        if (not transposed_flag) and "original" not in self.graph_data[
            "separate"
        ]["coo"]:
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )

        sub_dict_name = "transposed" if transposed_flag else "original"

        data_dict = dict()
        print("num relations in get_dgl_graph():", self.get_num_rels())
        if (
            "legacy_metadata_from_dgl" not in self.graph_data
            or "canonical_etypes"
            not in self.graph_data["legacy_metadata_from_dgl"]
        ):
            print(
                "WARNING: legacy_metadata_from_dgl not found, assuming only"
                " one node type in get_dgl_graph()"
            )
        for etype_idx in range(self.get_num_rels()):
            if (
                "legacy_metadata_from_dgl" not in self.graph_data
                or "canonical_etypes"
                not in self.graph_data["legacy_metadata_from_dgl"]
            ):
                curr_etype_canonical_name = (0, etype_idx, 0)
            else:
                curr_etype_canonical_name = self["legacy_metadata_from_dgl"][
                    "canonical_etypes"
                ][etype_idx]
            if transposed_flag:
                row_indices = self.graph_data["separate"]["coo"][
                    sub_dict_name
                ]["row_indices"][
                    self.graph_data["separate"]["coo"][sub_dict_name][
                        "rel_ptrs"
                    ][etype_idx] : self.graph_data["separate"]["coo"][
                        sub_dict_name
                    ][
                        "rel_ptrs"
                    ][
                        etype_idx + 1
                    ]
                ]
                col_indices = self.graph_data["separate"]["coo"][
                    sub_dict_name
                ]["col_indices"][
                    self.graph_data["separate"]["coo"][sub_dict_name][
                        "rel_ptrs"
                    ][etype_idx] : self.graph_data["separate"]["coo"][
                        sub_dict_name
                    ][
                        "rel_ptrs"
                    ][
                        etype_idx + 1
                    ]
                ]
                data_dict[curr_etype_canonical_name] = (
                    col_indices,
                    row_indices,
                )

        return dgl.heterograph(data_dict)

    @torch.no_grad()
    def calc_node_type_offset_from_dgl_heterograph(self, dglgraph):
        assert "original" in self.graph_data, "original not exists"
        self["original"]["node_type_offsets"] = torch.zeros(
            (len(dglgraph.ntypes) + 1,), dtype=torch.int64
        )
        for i in range(len(dglgraph.ntypes)):
            self["original"]["node_type_offsets"][i + 1] = self["original"][
                "node_type_offsets"
            ][i] + dglgraph.number_of_nodes(dglgraph.ntypes[i])
        self["original"]["node_type_offsets"] = self["original"][
            "node_type_offsets"
        ].to(self["original"]["eids"].device)

    def get_device(self):
        if "original" in self:
            return self["original"]["eids"].device
        elif "transposed" in self:
            return self["transposed"]["eids"].device
        else:
            raise ValueError(
                "Missing original or transposed data in graph_data"
            )
