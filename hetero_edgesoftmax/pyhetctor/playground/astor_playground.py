#!/usr/bin/env python3
import ast
import astor


expr0 = """
for e in g.Edges():
    z_i=W[e.etype]*e.src.feature
    z_j=W[e.etype]*e.dst.feature
    e.attn=leakyrelu(attn_vector[e.etype]*concat([z_i,z_j]))
for n in g.Nodes():
    n.softmax_sum = 0.0
    for e_incoming in IncomingEdges(n):
        n.softmax_sum+=exp(e_incoming.attn)
for e in g.Edges():
    e.attn = e.attn/e.dst.softmax_sum
    """
expr_ast0 = ast.parse(expr0)
# wrap each for loop with a new for loop
for node in ast.walk(expr_ast0):
    if isinstance(node, ast.For):
        node.body = [
            ast.For(target=node.target, iter=node.iter,
                    body=node.body, orelse=[])
        ]
        node.iter = ast.Name(id="range(1)")
        node.target = ast.Name(id="_")
# do a loop order change whenever it is two-level
for node in ast.walk(expr_ast0):
    if isinstance(node, ast.For):
        if len(node.body) == 1 and isinstance(node.body[0], ast.For):
            node.body[0].body, node.body[0].orelse = (
                node.body[0].orelse,
                node.body[0].body,
            )
# do a loop tiling if it is for e in g.Edges()
for node in ast.walk(expr_ast0):
    if isinstance(node, ast.For):
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Attribute)
            and node.iter.func.value.id == "g"
            and node.iter.func.attr == "Edges"
        ):
            # split it to two-level for loops where the first level is for n in g.Nodes() and the second level is for e in n.out_edges()
            node.iter = ast.Call(
                func=ast.Attribute(value=ast.Name(id="g"), attr="Nodes"),
                args=[],
                keywords=[],
            )  # ast.Call(func=ast.Name(id="Nodes"), args=[], keywords=[])
            old_body = node.body
            new_for = ast.For(
                target=ast.Name(id="e"),
                iter=ast.Call(
                    func=ast.Attribute(value=ast.Name(
                        id="n"), attr="out_edges"),
                    args=[],
                    keywords=[],
                ),
                body=[],
                orelse=[],
            )
            new_for.body = old_body

            node.body = []
            node.body.append(new_for)
            node.body[0].iter = ast.Call(
                func=ast.Attribute(value=ast.Name(id="n"), attr="out_edges"),
                args=[],
                keywords=[],
            )


print(astor.to_source(expr_ast0))
print(ast.dump(expr_ast0))
pass
