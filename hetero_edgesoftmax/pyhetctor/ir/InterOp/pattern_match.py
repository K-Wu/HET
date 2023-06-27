#!/usr/bin/env python3
import ast


# a list contains all matcher functions defined in this file
matchers = []

# TODO: there might be a chain of operations in one line, e.g.,
# n.output = n.feature * transpose(W)
# TO deal with this, we need to do the match in the following steps after canonicalization pass
# 1. match assignment. This helps us figure out the output
# 2. recursively match the right hand side of the assignment, where the right hand side entry function will be finally called at the end of any match function, i.e., after all other non-chain match logic failed
# note that upon entering the right-hand-side matching function, pass in 1) left-hand side results, and 2) for-loop levels. During recursive call, we can pass in the temporary result name instead. For example.
# when matchiing n.output = n.feature * transpose(W), the first time right-hand-side match is called, "node-wise iteration" and "n.output" are passed in. The second time right-hand-side match is called, "node-wise iteration" and "n.output_tmp1" are passed in


import ast

# import astor


def match_loop_nest_and_result(loop_root_node) -> list[OpBase]:
    """
    this is the entry point for pattern matchers. It figures out the for loop type and passes the return variable to the corresponding right-hand-side-only matchers.
    """
    # step 1 find the statement node inside loop nest and determine the loop-nest type
    assert isinstance(loop_root_node, ast.For)
    curr_node = loop_root_node
    loop_nest = []
    while isinstance(curr_node, ast.For):
        assert isinstance(curr_node.target, ast.Name)
        assert isinstance(curr_node.iter, ast.Call)
        assert isinstance(curr_node.iter.func, ast.Attribute)
        assert isinstance(curr_node.iter.func.value, ast.Name)
        loop_nest.append(
            (
                curr_node.target.id,
                curr_node.iter.func.value.id,
                curr_node.iter.func.attr,
            )
        )
        curr_node = curr_node.body[0]
    # determine the loop nest type
    loop_type = [
        item[2] for item in loop_nest
    ]  # types in examples involve ["nodes"], ["dst_nodes"], ["dst_nodes", "incoming_edges"], ["edges"]
    # TODO: pass the loop type to right-hand-side match function calls
    # step 2: match the result
    output_symbol = match_sole_return_value(curr_node)
    if output_symbol is None:
        output_symbol = match_dual_return_values(curr_node)
        if output_symbol is None:
            return None
        # match splitOp
        return match_unary_functions(output_symbol, loop_type, curr_node.value)
    match_nonlinear(output_symbol, loop_type, curr_node.value)
    match_copy_and_negation(output_symbol, loop_type, curr_node.value)
    match_unary_functions([output_symbol], loop_type, curr_node.value)
    match_node_linear(output_symbol, loop_type, curr_node.value)
    match_zero_initialize(output_symbol, loop_type, curr_node.value)
    match_node_multiplication(output_symbol, loop_type, curr_node.value)
    # TODO: do all matchers until there is a match


def match_node_multiplication(
    target: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    """
    this function matches node-wise multiplication operations
    like:
    for n in g.nodes():
        n["hs"] = V[n.ntype] * n.feature

    or
    for n in g.nodes():
        n["hs"] = linear(V[n.ntype], n.feature)
    """
    # if not isinstance(node, ast.Assign):
    #    return None
    # if len(node.targets) != 1:
    #    return None
    # target = node.targets[0]
    # if not isinstance(target, ast.Attribute):
    #     return False
    # if target.attr != "hs":
    #     return False
    if not isinstance(rhs_node, ast.BinOp):
        return None
    if rhs_node.op != ast.Mult:
        return None
    if not isinstance(rhs_node.left, ast.Name):
        return None
    if rhs_node.left.id != "n":
        # weight is on the left
        # TODO
        raise NotImplementedError

    else:
        # weight is on the right
        # TODO
        raise NotImplementedError

    return True


def match_weight_var(node):
    if isinstance(node, ast.Subscript):
        # sliced weights, e.g., V[n.ntype]
        print("Weight: ", node.value.id, node.slice.value.id, node.slice.attr)
        return True
    elif isinstance(node, ast.Name):
        print("Weight(Sliceless): ", node.id)
        return True
    return False


def match_edge_input(node):
    if isinstance(node, ast.Attribute):
        if node.value.id == "e":
            print("(EDGEWISE) input_key: ", node.attr)
            return True
    return False


# TODO: distinguish edgewise (g.edges()), from (n.incoming_edges())
def match_edge_output(node):
    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        if node.value.id == "e":
            print("(EDGEWISE) output_key: ", node.value.id, node.slice.value)
            return True
    return False


# TODO: lower xxx = 0.0 to an accumulation node or an assertion
def match_zero_initialize(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    # if not isinstance(node, ast.Assign):
    #    return None
    # output_symbol = match_sole_return_value(node)
    # if output_symbol is None or (not output_symbol.is_vertex_output()):
    #    return None
    if isinstance(rhs_node, ast.Constant):
        if rhs_node.value == 0.0:
            print(output_symbol, "is an accumulation node")
            return True
    return None


# TODO: distinguish nodewise (g.nodes()), from dst-node (g.dst_nodes())
def match_vertex_input(node):
    if isinstance(node, ast.Attribute):
        if node.value.id == "n":
            print("(NODEWISE) input_key: ", node.attr)
            return True
    return False


def match_vertex_output(node):
    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        if node.value.id == "n":
            print("(NODEWISE) output_key: ", node.value.id, node.slice.value)
            return True
    return False


def _match_return_values(node, num_return_values):
    if len(node.targets) != num_return_values:
        print(node.targets)
        return None
    output_symbols = []
    for idx_target in range(len(node.targets)):
        target = node.targets[idx_target]
        output_symbol = match_vertex_output(target)
        if output_symbol is None:
            output_symbol = match_edge_output(target)
            # if output_symbol is None:
            #    return None
        output_symbols.append(output_symbol)

    return output_symbols


def match_dual_return_values(node):
    output_symbols = _match_return_values(node, 2)
    if output_symbols[0] is None or output_symbols[1] is None:
        return None
    return output_symbols


def match_sole_return_value(node):
    output_symbols = _match_return_values(node, 1)
    return output_symbols[0]


# TODO: return None instead of False
# TODO:  generalize this function to match_edge_linear
# TODO: distinguish matmul, inner_product, outer_product by shape inference
def match_node_linear(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    """
    this function matches node-wise linear operations
    like:
    n["hs"] = V[n.ntype] * n.feature
    n["hs"] = linear(V[n.ntype], n.feature)
    """
    # TODO: check scope is foreach node iteration
    # if not isinstance(node, ast.Assign):
    #    return False
    # output_symbol = match_sole_return_value(node)
    # TODO: implement is_vertex_output(output_symbol)
    # if output_symbol is None:# or (not output_symbol.is_vertex_output()):
    #    return False
    if isinstance(rhs_node, ast.Call):
        if rhs_node.func.id != "linear":
            return False
        if len(rhs_node.args) != 2:
            return False
        if rhs_node.args[0].value.id != "n" and rhs_node.args[1].value.id != "n":
            return False

        if rhs_node.args[0].value.id == "n":
            assert match_vertex_input(rhs_node.args[0])
            assert match_weight_var(rhs_node.args[1])
            return True
        else:
            assert match_vertex_input(rhs_node.args[1])
            assert match_weight_var(rhs_node.args[0])
            return True
    elif isinstance(rhs_node, ast.BinOp):
        if not isinstance(rhs_node.op, ast.Mult):
            return False
        if rhs_node.left.value.id != "n" and rhs_node.right.value.id != "n":
            return False
        if rhs_node.left.value.id == "n":
            assert match_vertex_input(rhs_node.left)
            assert match_weight_var(rhs_node.right)
            return True
        else:
            assert match_vertex_input(rhs_node.right)
            assert match_weight_var(rhs_node.left)
            return True
    return False


# TODO: implement this
# TODO: this function matches transpose, concatenation and split
def match_unary_functions(
    output_symbols: list[VarBase], loop_type: list[tuple[str, str, str]], rhs_node
):
    # if not isinstance(node, ast.Assign):
    #    return None
    # if len(node.targets) != 1:
    if len(output_symbols) != 1:
        # should be split
        # output_symbols = match_dual_return_values(node)
        # if output_symbols is None:
        #    return None
        assert output_symbols is not None
        assert isinstance(rhs_node, ast.Call)
        assert rhs_node.func.id == "split"
    else:
        # output_symbol = match_sole_return_value(node)
        output_symbol = output_symbols[0]
        if output_symbol is None:
            return None
        if isinstance(rhs_node, ast.Call):
            if rhs_node.func.id == "transpose":
                raise NotImplementedError
            elif rhs_node.func.id == "concatenate":
                assert isinstance(rhs_node.args[0], ast.List)
                assert isinstance(rhs_node.args[0].elts[0], ast.Attribute)
                print(rhs_node.args[0].elts[0].value.id, rhs_node.args[0].elts[0].attr)
                assert isinstance(rhs_node.args[0].elts[1], ast.Attribute)
                print(rhs_node.args[0].elts[1].value.id, rhs_node.args[0].elts[1].attr)
    return None


# TODO: match negation
# TODO: return None instead of False
# This function matches copy, negation
def match_copy_and_negation(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    # if not isinstance(node, ast.Assign):
    #    return False
    # output_symbol = match_sole_return_value(node)
    # if output_symbol is None:
    #    return None
    input_symbol = match_vertex_input(rhs_node)
    if input_symbol is None:
        input_symbol = match_edge_input(rhs_node)
        if input_symbol is None:
            return False
    print(input_symbol, output_symbol)
    # TODO: create and return Copy object
    return True


# TODO: return None instead of False
def match_nonlinear(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    # if not isinstance(node, ast.Assign):
    #    return False
    # output_symbol = match_sole_return_value(node)
    # if output_symbol is None:
    #    return None
    if not isinstance(rhs_node, ast.Call):
        return False
    if not rhs_node.func.id in ["exp", "tanh"]:
        return False
    # todo: incorporate single argument function, i.e., concatenate, here
    if len(rhs_node.args) != 1:
        return False
    print("func name, ", rhs_node.func.id)
    print("input, ", rhs_node.args[0])
    # TODO: create and return NonLinear object
    return True


if __name__ == "__main__":
    print(ast.dump(ast.parse('e["zizj"] = 0.0')))
    print(ast.dump(ast.parse('e["zizj"] = concat([e.zi, e.zj])')))
    print(ast.dump(ast.parse('n["hs"] = Linear(V[n.ntype], n.feature)')))
    print(ast.dump(ast.parse('n["hs"] = V[n.ntype] * n.feature')))
    # print(match_node_linear(ast.parse("n[\"hs\"] = Linear(V[n.ntype], n.feature)").body[0]))
    # print(match_node_linear(ast.parse("n[\"hs\"] = V[n.ntype] * n.feature").body[0]))
    print(
        match_loop_nest_and_result(
            ast.parse(
                """for n in g.nodes():
        n[\"hs\"] = Linear(V[n.ntype], n.feature)"""
            ).body[0]
        )
    )
    print(
        match_loop_nest_and_result(
            ast.parse(
                """for n in g.nodes():
        n[\"hs\"] = V[n.ntype] * n.feature"""
            ).body[0]
        )
    )
