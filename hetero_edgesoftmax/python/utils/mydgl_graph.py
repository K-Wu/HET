#!/usr/bin/env python3


# NB: this class stores type in list and assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that.
from .. import kernels as K
from .. import utils

#    convert_integrated_csr_to_separate_csr,
#    convert_integrated_csr_to_separate_coo,
#    convert_integrated_coo_to_separate_csr,
#    convert_integrated_coo_to_separate_coo,
#    transpose_csr

import torch

import dgl


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
        if "row_ptr" in self.graph_data[original_or_transpose]:
            return "csr"
        elif "row_idx" in self.graph_data[original_or_transpose]:
            assert (
                "col_idx" in self.graph_data[original_or_transpose]
            ), "col_idx not exists"
            return "coo"
        else:
            raise ValueError("unknown sparse format")

    @torch.no_grad()
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
            ) = K.transpose_csr(
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

    @torch.no_grad()
    def generate_separate_csr_adj_for_each_etype(
        self, transposed_flag, sorted_rel_eid_flag=True
    ):
        # store rel_ptr, row_ptr, col_idx, eids in self.graph_data["separate"]["csr"]["original"] and self.graph_data["separate"]["csr"]["transposed"]
        if sorted_rel_eid_flag:
            assert (
                0
            ), "WARNING: sorted_rel_eid_flag is True, which is not supported for separate_csr_adj generation."
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

        # make sure the corresponding key in graph data is assigned an empty container
        if transposed_flag:
            if "transposed" not in self.graph_data["separate"]["csr"]:
                self.graph_data["separate"]["csr"]["transposed"] = dict()
            else:
                print(
                    "WARNING: in generating_separate transposed already exists, will be overwritten"
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
                    "WARNING : in generating_separate original already exists, will be overwritten"
                )

        original_or_transposed = "transposed" if transposed_flag else "original"

        # then call C++ wrapper function to do the job
        if self.get_sparse_format() == "csr":
            (
                separate_csr_rel_ptr,
                separate_csr_row_ptr,
                separate_csr_col_idx,
                separate_csr_eids,
            ) = K.convert_integrated_csr_to_separate_csr(
                self.graph_data[original_or_transposed]["row_ptr"],
                self.graph_data[original_or_transposed]["col_idx"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        elif self.get_sparse_format() == "coo":
            (
                separate_csr_rel_ptr,
                separate_csr_row_ptr,
                separate_csr_col_idx,
                separate_csr_eids,
            ) = K.convert_integrated_coo_to_separate_csr(
                self.graph_data[original_or_transposed]["row_idx"],
                self.graph_data[original_or_transposed]["col_idx"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        else:
            raise ValueError("unknown sparse format")
        self.graph_data["separate"]["csr"][original_or_transposed][
            "rel_ptr"
        ] = separate_csr_rel_ptr
        self.graph_data["separate"]["csr"][original_or_transposed][
            "row_ptr"
        ] = separate_csr_row_ptr
        self.graph_data["separate"]["csr"][original_or_transposed][
            "col_idx"
        ] = separate_csr_col_idx
        self.graph_data["separate"]["csr"][original_or_transposed][
            "eids"
        ] = separate_csr_eids

    @torch.no_grad()
    def generate_separate_coo_adj_for_each_etype(
        self, transposed_flag, rel_eid_sorted_flag=True
    ):
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
                    "WARNING : in generating_separate original already exists, will be overwritten"
                )

        original_or_transposed = "transposed" if transposed_flag else "original"
        # then call C++ wrapper function to do the job
        if self.get_sparse_format() == "csr":
            (
                separate_coo_rel_ptr,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = K.convert_integrated_csr_to_separate_coo(
                self.graph_data[original_or_transposed]["row_ptr"],
                self.graph_data[original_or_transposed]["col_idx"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        elif self.get_sparse_format() == "coo":
            (
                separate_coo_rel_ptr,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = K.convert_integrated_coo_to_separate_coo(
                self.graph_data[original_or_transposed]["row_idx"],
                self.graph_data[original_or_transposed]["col_idx"],
                self.graph_data[original_or_transposed]["rel_types"],
                self.graph_data[original_or_transposed]["eids"],
            )
        else:
            raise ValueError("unknown sparse format")

        if rel_eid_sorted_flag:
            # sort the coo by rel_type and eid
            (
                separate_coo_rel_ptr,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            ) = utils.sort_coo_by_etype_eids_torch_tensors(
                separate_coo_rel_ptr,
                separate_coo_row_idx,
                separate_coo_col_idx,
                separate_coo_eids,
            )
        self.graph_data["separate"]["coo"][original_or_transposed][
            "rel_ptr"
        ] = separate_coo_rel_ptr
        self.graph_data["separate"]["coo"][original_or_transposed][
            "row_idx"
        ] = separate_coo_row_idx
        self.graph_data["separate"]["coo"][original_or_transposed][
            "col_idx"
        ] = separate_coo_col_idx
        self.graph_data["separate"]["coo"][original_or_transposed][
            "eids"
        ] = separate_coo_eids

    @torch.no_grad()
    def generate_separate_adj_for_each_etype(
        self, separate_sparse_format, transposed_flag
    ):
        if separate_sparse_format == "csr":
            self.generate_separate_csr_adj_for_each_etype(transposed_flag)
        elif separate_sparse_format == "coo":
            self.generate_separate_coo_adj_for_each_etype(transposed_flag)
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def get_separate_node_idx_for_each_etype(self):
        if (
            "separate" not in self.graph_data
            or "coo" not in self.graph_data["separate"]
        ):
            raise ValueError(
                "separate coo graph data not found, please generate it first"
            )
        result_node_idx = torch.empty([0], dtype=torch.int64)
        result_rel_ptr = torch.empty([0], dtype=torch.int64)
        for idx_relation in range(self.get_num_rels()):
            node_idx_for_curr_relation = torch.unique(
                torch.concat(
                    [
                        self.graph_data["separate"]["coo"]["original"]["row_idx"][
                            self.graph_data["separate"]["coo"]["original"]["row_ptr"][
                                idx_relation
                            ] : self.graph_data["separate"]["coo"]["original"][
                                "row_ptr"
                            ][
                                idx_relation + 1
                            ]
                        ],
                        self.graph_data["separate"]["coo"]["original"]["col_idx"][
                            self.graph_data["separate"]["coo"]["original"]["row_ptr"][
                                idx_relation
                            ] : self.graph_data["separate"]["coo"]["original"][
                                "row_ptr"
                            ][
                                idx_relation + 1
                            ]
                        ],
                    ]
                )
            )

            result_node_idx = torch.cat([result_node_idx, node_idx_for_curr_relation])
            result_rel_ptr = torch.cat(
                [
                    result_rel_ptr,
                    (result_rel_ptr[-1] + node_idx_for_curr_relation.shape[0]),
                ]
            )

        if "unique_node_idx" in self.graph_data["separate"]:
            print("WARNING: unique_node_idx already exists, will be overwritten")
        self.graph_data["separate"]["unique_node_idx"] = dict()
        self.graph_data["separate"]["unique_node_idx"]["node_idx"] = result_node_idx
        self.graph_data["separate"]["unique_node_idx"]["rel_ptr"] = result_rel_ptr

    def import_metadata_from_dgl_heterograph(self, dglgraph):
        assert (
            "legacy_metadata_from_dgl" not in self.graph_data
        ), "legacy_metadata_from_dgl already exists"
        self["legacy_metadata_from_dgl"] = dict()
        self["legacy_metadata_from_dgl"]["canonical_etypes"] = dglgraph.canonical_etypes
        self["legacy_metadata_from_dgl"]["ntypes"] = dglgraph.ntypes
        self["legacy_metadata_from_dgl"]["number_of_nodes"] = dglgraph.number_of_nodes()
        self["legacy_metadata_from_dgl"]["number_of_nodes_per_type"] = dict(
            [(ntype, dglgraph.number_of_nodes(ntype)) for ntype in dglgraph.ntypes]
        )
        self["legacy_metadata_from_dgl"]["number_of_edges"] = dglgraph.number_of_edges()
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
            zip(dglgraph.canonical_etypes, range(len(dglgraph.canonical_etypes)))
        )

    def get_dgl_graph(self, transposed_flag: bool = False):
        if (
            "separate" not in self.graph_data
            or "coo" not in self.graph_data["separate"]
        ):
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )
        if transposed_flag and "transposed" not in self.graph_data["separate"]["coo"]:
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )
        if (not transposed_flag) and "original" not in self.graph_data["separate"][
            "coo"
        ]:
            self.generate_separate_coo_adj_for_each_etype(
                transposed_flag=transposed_flag
            )

        sub_dict_name = "transposed" if transposed_flag else "original"

        data_dict = dict()
        for etype_idx in range(self.get_num_rels()):
            if (
                "legacy_metadata_from_dgl" not in self.graph_data
                or "canonical_etypes" not in self.graph_data["legacy_metadata_from_dgl"]
            ):
                print(
                    "WARNING: legacy_metadata_from_dgl not found, assuming only one node type in get_dgl_graph()"
                )
                curr_etype_canonical_name = (0, etype_idx, 0)
            else:
                curr_etype_canonical_name = self["legacy_metadata_from_dgl"][
                    "canonical_etypes"
                ][etype_idx]
            if transposed_flag:
                row_indices = self.graph_data["separate"]["coo"][sub_dict_name][
                    "row_idx"
                ][
                    self.graph_data["separate"]["coo"][sub_dict_name]["rel_ptr"][
                        etype_idx
                    ] : self.graph_data["separate"]["coo"][sub_dict_name]["rel_ptr"][
                        etype_idx + 1
                    ]
                ]
                col_indices = self.graph_data["separate"]["coo"][sub_dict_name][
                    "col_idx"
                ][
                    self.graph_data["separate"]["coo"][sub_dict_name]["rel_ptr"][
                        etype_idx
                    ] : self.graph_data["separate"]["coo"][sub_dict_name]["rel_ptr"][
                        etype_idx + 1
                    ]
                ]
                data_dict[curr_etype_canonical_name] = (col_indices, row_indices)

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
            raise ValueError("Missing original or transposed data in graph_data")
