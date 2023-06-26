#!/usr/bin/env python3
"""
This file contains the APIs that mimic those in DGL and PyG. The goal is to support existing models with minimal changes.
The APIs could be implemented using frontend_interface or inter_op_ir, and utilties to switch the existing script to use our backend could also be provided in this file.
This file uses the techniques in [hetero_edgesoftmax/misc/playground/test_reroute_decorator.py]
"""
