import torch
import torch.jit
from typing import Union


@torch.jit.script
class MyScriptClass(object):
    def __init__(
        self,
        num_nodes: int,
        num_ntypes: int,
        num_rels: int,
        num_edges: int,
        sparse_format: str,
        transpose_format: Union[None, str],
    ):
        # self.graph_data = dict()
        self.num_nodes = num_nodes
        self.num_ntypes = num_ntypes
        self.num_rels = num_rels
        self.num_edges = num_edges
        self.sparse_format = sparse_format
        self.transposed_sparse_format = transpose_format


if __name__ == "__main__":
    mycls_obj = MyScriptClass(10, 2, 3, 20, "coo", None)
    print("nothing to do")
