#!/usr/bin/env python3

import torch
import torch.jit


from typing import Dict, Union


# restricted mydglgraph object. no device transfer support. Assuming the data are ready on the desired device during creation from mydglgraph object.

# metadata
# the mydglgraph object needs to execute get_num_nodes(), get_num_ntypes(), get_sparse_format(both original and transposed), get_device, already
# original needs to be existent in the mydglgraph object. Transposition and unique are also required if needed
# Data:
# self.graph_data["original"], as Dict[str, torch.Tensor], should exist
# optional: self.graph_data["transposed"], self.graph_data["separate"]["csr"]["original"], self.graph_data["separate"]["csr"]["transposed"], self.graph_data["separate"]["coo"]["original"], self.graph_data["separate"]["coo"]["transposed"], self.graph_data["separate"]["unique_node_idx"], all as Dict[str, torch.Tensor]
@torch.jit.script
class ScriptedMyDGLGraph(object):
    def __init__(
        self,
        num_nodes: int,
        num_ntypes: int,
        num_rels: int,
        num_edges: int,
        sparse_format: str,
        transposed_sparse_format: Union[None, str],
        original_coo: Union[None, Dict[str, torch.Tensor]],
        transposed_coo: Union[None, Dict[str, torch.Tensor]],
        in_csr: Union[None, Dict[str, torch.Tensor]],
        out_csr: Union[None, Dict[str, torch.Tensor]],
        original_node_type_offsets: Union[None, torch.Tensor],
        separate_unique_node_indices: Union[None, Dict[str, torch.Tensor]],
        separate_coo_original: Union[None, Dict[str, torch.Tensor]],
    ):
        # self.graph_data = dict()
        self.num_nodes = num_nodes
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels
        self.num_edges = num_edges
        self.sparse_format = sparse_format
        self.transposed_sparse_format = transposed_sparse_format
        self.original_coo = original_coo
        self.transposed_coo = transposed_coo
        self.in_csr = in_csr
        self.out_csr = out_csr
        self.original_node_type_offsets = original_node_type_offsets
        self.separate_unique_node_indices = separate_unique_node_indices
        self.separate_coo_original = separate_coo_original

    def get_num_nodes(self) -> int:
        return self.num_nodes

    def get_num_ntypes(self) -> int:
        return self.num_ntypes

    def get_num_rels(self) -> int:
        return self.num_rels

    # def get_device(self):
    #     return self.device

    def get_sparse_format(self, transpose_flag: bool = False) -> str:
        if transpose_flag:
            result = self.transposed_sparse_format
            assert result is not None
            return result
        else:
            return self.sparse_format

    def get_num_edges(self) -> int:
        return self.num_edges

    # def __setitem__(self, key, value):
    #    self.graph_data[key] = value

    # def __getitem__(self, key):
    #    return self.graph_data[key]

    # def __contains__(self, key):
    #    return key in self.graph_data

    def get_original_coo(self) -> Dict[str, torch.Tensor]:
        # G["original"]["rel_types"]
        # G["original"]["row_idx"]
        # G["original"]["col_idx"]
        # G["original"]["eids"]
        result = self.original_coo
        assert result is not None
        return result

    def get_transposed_coo(self) -> Dict[str, torch.Tensor]:
        # G["transposed"]["rel_types"]
        # G["transposed"]["row_idx"]
        # G["transposed"]["col_idx"]
        # G["transposed"]["eids"]
        result = self.transposed_coo
        assert result is not None
        return result

    def get_out_csr(self) -> Dict[str, torch.Tensor]:
        # G["original"]["rel_types"]
        # G["original"]["row_ptr"]
        # G["original"]["col_idx"]
        # G["original"]["eids"]
        result = self.out_csr
        assert result is not None
        return result

    def get_in_csr(self) -> Dict[str, torch.Tensor]:
        # G["transposed"]["rel_types"]
        # G["transposed"]["row_ptr"]
        # G["transposed"]["col_idx"]
        # G["transposed"]["eids"]

        # strip away Optional from http://zh0ngtian.tech/posts/3f804c9b.html
        result = self.in_csr
        assert result is not None
        return result

    def get_original_node_type_offsets(self) -> torch.Tensor:
        # G["original"]["node_type_offsets"]
        result = self.original_node_type_offsets
        assert result is not None
        return result

    def get_separate_unique_node_indices(self) -> Dict[str, torch.Tensor]:
        # G["separate"]["unique_node_idx"]["rel_ptr"],
        # G["separate"]["unique_node_idx"]["node_idx"],
        result = self.separate_unique_node_indices
        assert result is not None
        return result

    def get_separate_coo_original(self) -> Dict[str, torch.Tensor]:
        # G["separate"]["coo"]["original"]["rel_ptr"],
        # G["separate"]["coo"]["original"]["row_idx"],
        # G["separate"]["coo"]["original"]["col_idx"],
        # G["separate"]["coo"]["original"]["eids"],

        result = self.separate_coo_original
        assert result is not None
        return result
