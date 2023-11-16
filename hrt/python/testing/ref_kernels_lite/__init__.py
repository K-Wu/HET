#!/usr/bin/env python3
from .ref_rgnn import *
from .ref_rgat import *
from .ref_rgcn import *
from .ref_hgt import *


import sys
import inspect
from functools import partial

COMPACT_FUNCTIONS = (
    towrap_relational_fused_gat_kernel_compact_as_of_node_separate_coo_dual_unique_node_list,
    towrap_backward_relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list,
    towrap_relational_fused_gat_kernel_compact_as_of_node_separate_coo,
    towrap_backward_relational_fused_gat_compact_as_of_node_separate_coo,
    towrap_rgnn_inner_product_node_compact_and_node,
    towrap_backward_rgnn_inner_product_node_compact_and_node,
)  # the first argument of functions with compaction is the compaction mapping


def set_compact_mapping(compact_mapping):
    # globals()
    # dir()
    # from https://stackoverflow.com/a/63413129/5555077
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if (
            inspect.isfunction(obj)
            and name in COMPACT_FUNCTIONS
            and obj.__module__ == __name__
        ):
            globals()[name] = partial(obj, compact_mapping)
            # or dir()
