#!/usr/bin/env python3
from .ref_rgnn import *
from .ref_rgat import *
from .ref_rgcn import *
from .ref_hgt import *


import sys
import inspect

COMPACT_FUNCTIONS = (
    []
)  # the first argument of functions with compaction is the compaction mapping


def set_compact_mapping(compact_mapping):
    raise NotImplementedError
    # globals()
    # dir()
    # from https://stackoverflow.com/a/63413129/5555077
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if (
            inspect.isfunction(obj)
            and name in COMPACT_FUNCTIONS
            and obj.__module__ == __name__
        ):
            dir()[name] = partial(obj, compact_mapping)
            # or globals()
