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
#    e["raw_attn_tmp"] = e.src.hs_attn * W_attn[e.etype]
# for e in g.edges():
#    e["raw_attn_tmp2"] = e.raw_attn_tmp * e.dst.ht_attn
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


# after loop canonicalization, every node in module body is a for-loop node, and every for-loop node has only one node in its body, whether it is mult-level for-loop or not
def canonicalize_for_loop_pass(module_node):
    new_body = []
    for node in module_node.body:
        if not isinstance(node, ast.For):
            new_body.append(node)
        # split for loop to single-statement loop nest
        if isinstance(node, ast.For):
            new_body += split_for_loop_node(node)
    module_node.body = new_body
    return module_node
