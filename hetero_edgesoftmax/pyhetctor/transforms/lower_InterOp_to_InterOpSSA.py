#!/usr/bin/env python3
from .InterOp import canonicalize
from .InterOp.pattern_match import matchers


# the input is canonicalized program, where each node in the module body is a single-statement loop-nest
# the procedure runs all matchers on all single-statement loop-nest nodes, and put the SSA statement into the returning list once there is a match
# return program at inter-op-ssa level
def lower_ops(module_node) -> list:
    ssa_statements = []
    for node in module_node.body:
        for matcher in matchers:
            ssa_statement = matcher(node)
            if ssa_statement is not None:
                ssa_statements.append(ssa_statement)
                break
    return ssa_statements


# TODO: add shape inference so that we return a complete inter_op_ssa_prog
def lower(module_node):
    module_node = canonicalize.canonicalize_for_loop_pass(module_node)
    ssa_statements = lower_ops(module_node)
    raise NotImplementedError("shape information is not available yet")
    return inter_op_ssa_prog
