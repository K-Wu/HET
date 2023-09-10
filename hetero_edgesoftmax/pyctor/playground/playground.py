#!/usr/bin/env python3
import ast
import astpretty


def example1():
    expr = """
    def add(arg1, arg2):
        return func(arg1) + arg2
    """
    expr_ast = ast.parse(expr)
    print(ast.dump(ast.parse(expr_ast)))


def test():
    for e in g.edges():
        zi = W[e.etype] * e.src.feature
        zj = W[e.etype] * e.dst.feature
        e["attn"] = leakyrelu(inner_prod(attn_vec[e.etype], concat([zi, zj])))
    for n in g.dest_nodes():
        n["softmax_sum"] = 0.0
        for e in n.incoming_edges():
            n["softmax_sum"] += exp(e["attn"])
    for e in g.edges():
        e["attn"] = e["attn"] / e.dst["softmax_sum"]


if __name__ == "__main__":
    # interactive playground: https://python-ast-explorer.com/
    # doc https://greentreesnakes.readthedocs.io/en/latest/
    expr1 = """
for e in Edges():
    z_i=W[e.etype]*e.src.feature
    z_j=W[e.etype]*e.dst.feature
    e.attn=leakyrelu(attn_vector[e.etype]*concat([z_i,z_j]))
    """
    expr2 = """
for n in Nodes():
    n.softmax_sum = 0.0
    for e_incoming in IncomingEdges(n):
        n.softmax_sum+=exp(e_incoming.attn)
    """

    expr0 = """
for e in Edges():
    z_i=W[e.etype]*e.src.feature
    z_j=W[e.etype]*e.dst.feature
    e.attn=leakyrelu(attn_vector[e.etype]*concat([z_i,z_j]))
for n in Nodes():
    n.softmax_sum = 0.0
    for e_incoming in IncomingEdges(n):
        n.softmax_sum+=exp(e_incoming.attn)
for e in Edges():
    e.attn = e.attn/e.dst.softmax_sum
    """
    expr_ast0 = ast.parse(expr0)
    astpretty.pprint(ast.parse(expr_ast0))
    pass
    import libcst as cst

    result = cst.parse_statement(expr1)
    result2 = cst.parse_statement(expr2)
    result0 = cst.parse_module(expr0)
    pass
