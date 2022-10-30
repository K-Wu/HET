#!/usr/bin/env python3


# NB: this class stores type in list and assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that.
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

    def cuda(self):
        for key in self.graph_data:
            if key == "legacy_metadata_from_dgl":
                continue
            for second_key in self.graph_data[key]:
                self.graph_data[key][second_key] = self.graph_data[key][
                    second_key
                ].cuda()

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
