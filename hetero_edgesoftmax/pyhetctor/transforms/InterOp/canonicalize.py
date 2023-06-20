#!/usr/bin/env python3
import ast

# This transform 1) breaks down operation chain to multiple operations, and 2) does loop split to faciliate scope analysis
# e.g.
# for e in g.edges():
# e["msg"] = e.src.hs * W_msg[e.etype]
# e["raw_attn"] = exp(e.src.hs_attn * W_attn[e.etype] * e.dst.ht_attn)
# =>
# for e in g.edges():
#    e["msg"] = e.src.hs * W_msg[e.etype]
# for e in g.edges():
#    e["raw_attn_tmp1"] = e.src.hs_attn * W_attn[e.etype]
# for e in g.edges():
#    e["raw_attn_tmp2"] = e.raw_attn_tmp1 * e.dst.ht_attn
# for e in g.edges():
#   e["raw_attn"] = exp(e.raw_attn_tmp2)


# return a list of for nodes: each is a single-statement loop nest where each statement is from the body of this for_node
def split_for_loop_node(for_node):
    results = []
    assert isinstance(for_node, ast.For)
    if len(for_node.orelse) != 0:
        raise NotImplementedError("unsupported else branch")
    for node in for_node.body:
        if not isinstance(node, ast.For):
            new_node = ast.For(
                target=for_node.target, iter=for_node.iter, body=[node], orelse=[]
            )
            results.append(new_node)
        else:
            results += split_for_loop_node(node)
    return results


# As an ad hoc solution, we assert the names are canonicalized rather than renaming it
# TODO: implement renaming if the name is not canonicalized when inputted
def assert_for_loop_variables_canonicalized(for_node):
    assert isinstance(for_node, ast.For)
    assert isinstance(for_node.iter, ast.Call)
    assert isinstance(for_node.iter.func, ast.Attribute)
    assert isinstance(for_node.iter.func.value, ast.Name)
    if isinstance(for_node.body[0], ast.For):
        assert_for_loop_variables_canonicalized(for_node.body[0])
    assert for_node.iter.func.value.id in ["g", "n"]
    if for_node.iter.func.attr in ["nodes", "dst_nodes", "src_nodes"]:
        assert for_node.target.id == "n"
        return
    elif for_node.iter.func.attr == ["edges", "incoming_edges", "outgoing_edges"]:
        assert for_node.target.id == "e"
        return
    else:
        raise ValueError("unrecognized iter function")


# after loop canonicalization, 1) every node in module body is a for-loop node, and every for-loop node has only one node in its body, whether it is mult-level for-loop or not
# 2) rename loop variables to n and e according to the loop iteration type
def canonicalize_for_loop_pass(module_node):
    new_body = []
    for node in module_node.body:
        if not isinstance(node, ast.For):
            new_body.append(node)
        # split for loop to single-statement loop nest
        if isinstance(node, ast.For):
            new_body += split_for_loop_node(node)
    for node in module_node.body:
        assert_for_loop_variables_canonicalized(node)
    module_node.body = new_body
    return module_node
