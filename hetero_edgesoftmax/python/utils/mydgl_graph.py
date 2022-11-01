#!/usr/bin/env python3


# NB: this class stores type in list and assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that.
from .sparse_matrix_converters import (
    convert_integrated_csr_to_separate_csr,
    convert_integrated_csr_to_separate_coo,
    convert_integrated_coo_to_separate_csr,
    convert_integrated_coo_to_separate_coo,
    transpose_csr,
)
import torch


class MyDGLGraph:
    # TODO: impl ["legacy_metadata_from_dgl"]["canonical_etypes"] elements are (srctype, etype, dsttype) in G.canonical_etypes
    # TODO: impl G["legacy_metadata_from_dgl"]["ntypes"] elements are in G.ntypes
    # TODO: impl G["legacy_metadata_from_dgl"]["number_of_nodes"] element is G.number_of_nodes()
    # TODO: impl G["original"]["node_type_offsets"]
    def __init__(self):
        import platform

        assert (
            platform.python_implementation() == "CPython"
        ), "This class assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that."
        self.graph_data = dict()

    def __setitem__(self, key, value):
        self.graph_data[key] = value

    def __getitem__(self, key):
        return self.graph_data[key]

    def __contains__(self, key):
        return key in self.graph_data

    def get_sparse_format(self):
        if "row_ptr" in self.graph_data["original"]:
            return "csr"
        elif "row_idx" in self.graph_data["original"]:
            assert "col_idx" in self.graph_data["original"], "col_idx not exists"
            return "coo"
        else:
            raise ValueError("unknown sparse format")

    def transpose(self):
        if self.get_sparse_format() == "csr":
            (
                self.graph_data["original"]["row_ptr"],
                self.graph_data["original"]["col_idx"],
            ) = (
                self.graph_data["original"]["col_idx"],
                self.graph_data["original"]["row_ptr"],
            )
        elif self.get_sparse_format() == "coo":
            assert "transposed" not in self.graph_data, "transposed already exists"
            self.graph_data["transposed"] = dict()
            self.graph_data["transposed"]["row_idx"] = self.graph_data["original"][
                "col_idx"
            ]
            self.graph_data["transposed"]["col_idx"] = self.graph_data["original"][
                "row_idx"
            ]
            self.graph_data["transposed"]["eids"] = self.graph_data["original"]["eids"]
            self.graph_data["transposed"]["rel_types"] = self.graph_data["original"][
                "rel_types"
            ]
        else:
            assert "transposed" not in self.graph_data, "transposed already exists"
            self.graph_data["transposed"] = dict()
            (
                transposed_rowptr,
                transposed_col_idx,
                transposed_rel_types,
                transposed_eids,
            ) = transpose_csr(
                self.graph_data["original"]["row_ptr"],
                self.graph_data["original"]["col_idx"],
                self.graph_data["original"]["rel_types"],
                self.graph_data["original"]["eids"],
            )
            self.graph_data["transposed"]["row_ptr"] = transposed_rowptr
            self.graph_data["transposed"]["col_idx"] = transposed_col_idx
            self.graph_data["transposed"]["rel_types"] = transposed_rel_types
            self.graph_data["transposed"]["eids"] = transposed_eids
        return self

    def generate_separate_csr_adj_for_each_etype(self, transposed_flag):
        # store rel_ptr, row_ptr, col_idx, eids in self.graph_data["separate"]["csr"]["original"] and self.graph_data["separate"]["csr"]["transposed"]

        # first make sure graph data, as torch tensors, are on cpu
        self = self.cpu()

        if "separate" not in self.graph_data:
            self.graph_data["separate"] = dict()
        else:
            print(
                "WARNING : in generating_separate separate already exists, will be overwritten"
            )
        if "csr" not in self.graph_data["separate"]:
            self.graph_data["separate"]["csr"] = dict()
        else:
            print(
                "WARNING : in generating_separate csr already exists, will be overwritten"
            )

        if transposed_flag:
            if "transposed" not in self.graph_data["separate"]["csr"]:
                self.graph_data["separate"]["csr"]["transposed"] = dict()
            else:
                print(
                    "WARNING: in generating_separate transposed already exists, will be overwritten"
                )
            # then call C++ wrapper function to do the job
            raise NotImplementedError(
                "convertion to transposed separate csr not implemented"
            )
        else:
            if "original" not in self.graph_data["separate"]["csr"]:
                self.graph_data["separate"]["csr"]["original"] = dict()
            else:
                print(
                    "WARNING : in generating_separate original already exists, will be overwritten"
                )
            # then call C++ wrapper function to do the job
            if self.get_sparse_format() == "csr":
                (
                    separate_csr_rel_ptr,
                    separate_csr_row_ptr,
                    separate_csr_col_idx,
                    separate_csr_eids,
                ) = convert_integrated_csr_to_separate_csr(
                    self.graph_data["original"]["row_ptr"],
                    self.graph_data["original"]["col_idx"],
                    self.graph_data["original"]["rel_types"],
                    self.graph_data["original"]["eids"],
                )
            elif self.get_sparse_format() == "coo":
                (
                    separate_csr_rel_ptr,
                    separate_csr_row_ptr,
                    separate_csr_col_idx,
                    separate_csr_eids,
                ) = convert_integrated_coo_to_separate_csr(
                    self.graph_data["original"]["row_idx"],
                    self.graph_data["original"]["col_idx"],
                    self.graph_data["original"]["rel_types"],
                    self.graph_data["original"]["eids"],
                )
            else:
                raise ValueError("unknown sparse format")
            self.graph_data["separate"]["csr"]["original"][
                "rel_ptr"
            ] = separate_csr_rel_ptr
            self.graph_data["separate"]["csr"]["original"][
                "row_ptr"
            ] = separate_csr_row_ptr
            self.graph_data["separate"]["csr"]["original"][
                "col_idx"
            ] = separate_csr_col_idx
            self.graph_data["separate"]["csr"]["original"]["eids"] = separate_csr_eids

    def generate_separate_coo_adj_for_each_etype(self, transposed_flag):
        # store rel_ptr, row_idx, col_idx, eids in self.graph_data["separate"]["coo"]["original"] and self.graph_data["separate"]["coo"]["transposed"]
        # first make sure graph data, as torch tensors, are on cpu
        self = self.cpu()

        if "separate" not in self.graph_data:
            self.graph_data["separate"] = dict()
        else:
            print(
                "WARNING : in generating_separate separate already exists, will be overwritten"
            )
        if "coo" not in self.graph_data["separate"]:
            self.graph_data["separate"]["coo"] = dict()
        else:
            print(
                "WARNING : in generating_separate coo already exists, will be overwritten"
            )

        if transposed_flag:
            if "transposed" not in self.graph_data["separate"]["coo"]:
                self.graph_data["separate"]["coo"]["transposed"] = dict()
            else:
                print(
                    "WARNING: in generating_separate transposed already exists, will be overwritten"
                )
            # then call C++ wrapper function to do the job
            raise NotImplementedError(
                "convertion to transposed separate coo not implemented"
            )
        else:
            if "original" not in self.graph_data["separate"]["coo"]:
                self.graph_data["separate"]["coo"]["original"] = dict()
            else:
                print(
                    "WARNING : in generating_separate original already exists, will be overwritten"
                )
            # then call C++ wrapper function to do the job
            if self.get_sparse_format() == "csr":
                (
                    separate_coo_rel_ptr,
                    separate_coo_row_idx,
                    separate_coo_col_idx,
                    separate_coo_eids,
                ) = convert_integrated_csr_to_separate_coo(
                    self.graph_data["original"]["row_ptr"],
                    self.graph_data["original"]["col_idx"],
                    self.graph_data["original"]["rel_types"],
                    self.graph_data["original"]["eids"],
                )
            elif self.get_sparse_format() == "coo":
                (
                    separate_coo_rel_ptr,
                    separate_coo_row_idx,
                    separate_coo_col_idx,
                    separate_coo_eids,
                ) = convert_integrated_coo_to_separate_coo(
                    self.graph_data["original"]["row_idx"],
                    self.graph_data["original"]["col_idx"],
                    self.graph_data["original"]["rel_types"],
                    self.graph_data["original"]["eids"],
                )
            else:
                raise ValueError("unknown sparse format")
            self.graph_data["separate"]["coo"]["original"][
                "rel_ptr"
            ] = separate_coo_rel_ptr
            self.graph_data["separate"]["coo"]["original"][
                "row_idx"
            ] = separate_coo_row_idx
            self.graph_data["separate"]["coo"]["original"][
                "col_idx"
            ] = separate_coo_col_idx
            self.graph_data["separate"]["coo"]["original"]["eids"] = separate_coo_eids

    def generate_separate_adj_for_each_etype(self, separate_sparse_format):
        if separate_sparse_format == "csr":
            self.generate_separate_csr_adj_for_each_etype()
        elif separate_sparse_format == "coo":
            self.generate_separate_coo_adj_for_each_etype()
        else:
            raise NotImplementedError()

    def import_metadata_from_dgl_heterograph(self, dglgraph):
        assert (
            "legacy_metadata_from_dgl" not in self.graph_data
        ), "legacy_metadata_from_dgl already exists"
        self["legacy_metadata_from_dgl"] = dict()
        self["legacy_metadata_from_dgl"]["canonical_etypes"] = dglgraph.canonical_etypes
        self["legacy_metadata_from_dgl"]["ntypes"] = dglgraph.ntypes
        self["legacy_metadata_from_dgl"]["number_of_nodes"] = dglgraph.number_of_nodes()
        self["legacy_metadata_from_dgl"]["node_dict"] = dict(
            zip(dglgraph.ntypes, range(len(dglgraph.ntypes)))
        )
        self["legacy_metadata_from_dgl"]["edge_dict"] = dict(
            zip(dglgraph.canonical_etypes, range(len(dglgraph.canonical_etypes)))
        )

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

    def to(self, device):
        for key in self.graph_data:
            if key == "legacy_metadata_from_dgl":
                continue
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][second_key].to(
                    device
                )
        return self

    def cuda(self):
        for key in self.graph_data:
            if key == "legacy_metadata_from_dgl":
                continue
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][
                    second_key
                ].cuda()
        return self

    def cpu(self):
        for key in self.graph_data:
            if key == "legacy_metadata_from_dgl":
                continue
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][
                    second_key
                ].cpu()
        return self

    def get_num_nodes(self):
        if "row_ptr" in self.graph_data["original"]:
            return self.graph_data["original"]["row_ptr"].numel() - 1
        else:
            assert "row_idx" in self.graph_data["original"], "row_idx not exists"
            assert "col_idx" in self.graph_data["original"], "col_idx not exists"
            return max(
                int(self.graph_data["original"]["row_idx"].max()) + 1,
                int(self.graph_data["original"]["col_idx"].max()) + 1,
            )

    def get_num_rels(self):
        return int(self.graph_data["original"]["rel_types"].max().item()) + 1

    def get_num_edges(self):
        return self.graph_data["original"]["rel_types"].numel()
