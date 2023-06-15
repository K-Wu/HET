#!/usr/bin/env python3
import ast
import astor

expr_l = """
for e in g.Edges():
    e.dst["sum1"] += e.data123
for e in g.Edges():
    e["data1"] = e.data2 / e.dst.sum1
"""
expr_l_ast = ast.parse(expr_l)
print(ast.dump(expr_l_ast))


def is_call_edges(call):
    if isinstance(call, ast.Call):
        f_func = call.func
        if isinstance(f_func, ast.Attribute) and isinstance(f_func.value, ast.Name):
            f_object = f_func.value.id
            f_method = f_func.attr
            if f_object == 'g' and f_method == 'Edges':
                return True

    return False


def dst_node_var(n, e):
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.value.id == e and n.value.attr == 'dst':
        return n.slice.value
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.value.id == e and n.value.attr == 'dst':
        return n.attr
    else:
        return None


global_vars = {}


for node in ast.walk(expr_l_ast):
    if isinstance(node, ast.For) and is_call_edges(node.iter) and isinstance(node.target, ast.Name):
        edge_var = node.target.id
        if len(node.body) == 1:
            body = node.body[0]
            if isinstance(body, ast.AugAssign) and isinstance(body.target, ast.Subscript):
                var_name = dst_node_var(body.target, edge_var)
                if var_name is not None:
                    if isinstance(body.value, ast.Attribute) and body.value.value.id == edge_var and body.value.attr != var_name:
                        global_vars[var_name] = [node]
                        body.target = ast.Name(id='n_var')
            elif isinstance(body, ast.Assign) and len(body.targets) == 1 and isinstance(body.targets[0], ast.Subscript):
                assign_var = dst_node_var(body.targets[0], edge_var)
                if assign_var not in global_vars:
                    if isinstance(body.value, ast.BinOp) and isinstance(body.value.left, ast.Attribute) and isinstance(
                            body.value.right, ast.Attribute):
                        var_left = dst_node_var(body.value.left, edge_var)
                        var_right = dst_node_var(body.value.right, edge_var)
                        if var_left in global_vars:
                            global_vars[var_left].append(node)
                            body.value.left = ast.Name(id='n_var')
                        elif var_right in global_vars:
                            global_vars[var_right].append(node)
                            body.value.right = ast.Name(id='n_var')


for var, nodes in global_vars.items():
    if len(nodes) == 2:
        re_ast = ast.For(
            target=ast.Name(id='n'),
            iter=ast.Call(func=ast.Attribute(value=ast.Name(id='g'), attr='Nodes'), args=[], keywords=[]),
            body=[
                ast.Assign(targets=[ast.Name(id='n_var')], value=ast.Constant(value=0.0)),
                ast.For(
                    target=ast.Name(id='e'),
                    iter=ast.Call(func=ast.Attribute(value=ast.Name(id='n'), attr='incoming_edges'), args=[], keywords=[]),
                    body=nodes[0].body,
                    orelse=[]
                ),
                ast.For(
                    target=ast.Name(id='e'),
                    iter=ast.Call(func=ast.Attribute(value=ast.Name(id='n'), attr='incoming_edges'), args=[], keywords=[]),
                    body=nodes[1].body,
                    orelse=[]
                )
            ],
            orelse=[]
        )
        expr_l_ast.body.append(re_ast)
        expr_l_ast.body.remove(nodes[0])
        expr_l_ast.body.remove(nodes[1])

print(astor.to_source(expr_l_ast))

# expr0 = """
# for e in g.Edges():
#     z_i=W[e.etype]*e.src.feature
#     z_j=W[e.etype]*e.dst.feature
#     e.attn=leakyrelu(attn_vector[e.etype]*concat([z_i,z_j]))
# for n in g.Nodes():
#     n.softmax_sum = 0.0
#     for e_incoming in IncomingEdges(n):
#         n.softmax_sum+=exp(e_incoming.attn)
# for e in g.Edges():
#     e.attn = e.attn/e.dst.softmax_sum
#     """
# expr_ast0 = ast.parse(expr0)
# wrap each for loop with a new for loop
# for node in ast.walk(expr_ast0):
#     if isinstance(node, ast.For):
#         node.body = [
#             ast.For(target=node.target, iter=node.iter, body=node.body, orelse=[])
#         ]
#         node.iter = ast.Name(id="range(1)")
#         node.target = ast.Name(id="_")
# # do a loop order change whenever it is two-level
# for node in ast.walk(expr_ast0):
#     if isinstance(node, ast.For):
#         if len(node.body) == 1 and isinstance(node.body[0], ast.For):
#             node.body[0].body, node.body[0].orelse = (
#                 node.body[0].orelse,
#                 node.body[0].body,
#             )
# # do a loop tiling if it is for e in g.Edges()
# for node in ast.walk(expr_ast0):
#     if isinstance(node, ast.For):
#         if (
#             isinstance(node.iter, ast.Call)
#             and isinstance(node.iter.func, ast.Attribute)
#             and node.iter.func.value.id == "g"
#             and node.iter.func.attr == "Edges"
#         ):
#             # split it to two-level for loops where the first level is for n in g.Nodes() and the second level is for e in n.out_edges()
#             node.iter = ast.Call(
#                 func=ast.Attribute(value=ast.Name(id="g"), attr="Nodes"),
#                 args=[],
#                 keywords=[],
#             )  # ast.Call(func=ast.Name(id="Nodes"), args=[], keywords=[])
#             old_body = node.body
#             new_for = ast.For(
#                 target=ast.Name(id="e"),
#                 iter=ast.Call(
#                     func=ast.Attribute(value=ast.Name(id="n"), attr="out_edges"),
#                     args=[],
#                     keywords=[],
#                 ),
#                 body=[],
#                 orelse=[],
#             )
#             new_for.body = old_body
#
#             node.body = []
#             node.body.append(new_for)
#             node.body[0].iter = ast.Call(
#                 func=ast.Attribute(value=ast.Name(id="n"), attr="out_edges"),
#                 args=[],
#                 keywords=[],
#             )


# print(astor.to_source(expr_ast0))
# print(ast.dump(expr_ast0))
pass
