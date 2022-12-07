#!/usr/bin/env python3
import torch

from .mydgl_graph import *
from .sparse_matrix_converters import *
from .loaders_from_npy import *
from .graphiler_datasets import *
from .coo_sorters import *
from .graphiler_bench import *
from .graph_synthesizers import *
from .graph_sampler import *
from .utils import *
from .mydglgraph_converters import *

import platform

assert (
    platform.python_implementation() == "CPython"
), "mydglgraph class assumes the type order in this class and dglgraph are preserved across run. Therefore one should use the CPython implementation to ensure that."
