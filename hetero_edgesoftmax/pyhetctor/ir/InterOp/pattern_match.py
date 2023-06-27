#!/usr/bin/env python3
import ast
from ..InterOpSSA.operators import OpBase
from ..InterOpSSA.operators import *
from ..InterOpSSA.variables import VarBase
from ..InterOpSSA.variables import *

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


def match_loop_nest_and_result(loop_root_node) -> Union[list[OpBase], None]:
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

    assert isinstance(curr_node, ast.Assign)
    # TODO: pass the loop type to right-hand-side match function calls
    # step 2: match the result
    output_symbol = match_sole_return_value(curr_node)
    if output_symbol is None:
        output_symbol = match_dual_return_values(curr_node)
        if output_symbol is None:
            return None
        # match splitOp
        return match_unary_functions(output_symbol, loop_type, curr_node.value)
    # TODO: make this recursive so that chain of operations on the right hand side can be matched
    ops = match_nonlinear(output_symbol, loop_type, curr_node.value)
    if ops:
        return ops
    ops = match_copy_and_negation(output_symbol, loop_type, curr_node.value)
    if ops:
        return ops
    ops = match_unary_functions([output_symbol], loop_type, curr_node.value)
    if ops:
        return ops
    ops = match_node_linear(output_symbol, loop_type, curr_node.value)
    if ops:
        return ops
    ops = match_zero_initialize(output_symbol, loop_type, curr_node.value)
    if ops:
        return ops
    ops = match_node_multiplication(output_symbol, loop_type, curr_node.value)
    if ops:
        return ops


# TODO: generalize this to handle edgewise as well
def match_node_multiplication(
    target: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
) -> Union[list[OpBase], None]:
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
        weight_symbol = match_weight_var(rhs_node.left)
        input_symbol = match_data_input(rhs_node.right)
    else:
        # weight is on the right
        weight_symbol = match_weight_var(rhs_node.right)
        input_symbol = match_data_input(rhs_node.left)
    if weight_symbol is None or input_symbol is None:
        return None
    # determine multiplication type
    raise NotImplementedError


def match_weight_var(node) -> Union[WeightVar, None]:
    if isinstance(node, ast.Subscript):
        # sliced weights, e.g., V[n.ntype]
        assert isinstance(node.value, ast.Name)
        assert isinstance(node.slice, ast.Attribute)
        assert isinstance(node.slice.value, ast.Name)
        print("Weight: ", node.value.id, node.slice.value.id, node.slice.attr)
        if node.slice.attr == "ntype":
            slice_type = "NODETYPE"
        elif node.slice.attr == "etype":
            slice_type = "EDGETYPE"
        return WeightVar.from_dict({"name": node.value.id, "slice_type": slice_type})
    elif isinstance(node, ast.Name):
        print("Weight(Sliceless): ", node.id)
        return WeightVar.from_dict({"name": node.id, "slice_type": "NONE"})
    return None


def match_edge_input(node) -> Union[DataVar, None]:
    if isinstance(node, ast.Attribute):
        assert isinstance(node.value, ast.Name)
        if node.value.id == "e":
            print("(EDGEWISE) input_key: ", node.attr)
            return DataVar.from_dict({"name": node.attr, "type": "EDGEWISE"})
    return None


# TODO: distinguish edgewise (g.edges()), from (n.incoming_edges())
def match_edge_output(node) -> Union[DataVar, None]:
    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        assert isinstance(node.value, ast.Name)
        if node.value.id == "e":
            assert isinstance(node.slice, ast.Constant)
            print("(EDGEWISE) output_key: ", node.value.id, node.slice.value)
            return DataVar.from_dict({"name": node.slice.value, "type": "EDGEWISE"})
    return None


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
            # TODO: return a hint or assertion
            raise NotImplementedError
    return None


# TODO: distinguish nodewise (g.nodes()), from dst-node (g.dst_nodes())
def match_vertex_input(node) -> Union[DataVar, None]:
    if isinstance(node, ast.Attribute):
        assert isinstance(node.value, ast.Name)
        if node.value.id == "n":
            print("(NODEWISE) input_key: ", node.attr)
            # distinguish DST_NODE, SRC_NODE from NODEWISE
            return DataVar.from_dict({"name": node.attr, "type": "NODEWISE"})
    return None


def match_data_input(node) -> Union[DataVar, None]:
    input_symbol = match_vertex_input(node)
    if input_symbol is None:
        return match_edge_input(node)
    return input_symbol


def match_vertex_output(node) -> Union[DataVar, None]:
    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        assert isinstance(node.value, ast.Name)
        if node.value.id == "n":
            assert isinstance(node.slice, ast.Constant)
            print("(NODEWISE) output_key: ", node.value.id, node.slice.value)
            return DataVar.from_dict({"name": node.slice.value, "type": "NODEWISE"})
    return None


def _match_return_values(node, num_return_values) -> list[Union[VarBase, None]]:
    if len(node.targets) != num_return_values:
        print(node.targets)
        return [None] * num_return_values
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


def match_dual_return_values(node) -> Union[list[VarBase], None]:
    output_symbols = _match_return_values(node, 2)
    if output_symbols[0] is None or output_symbols[1] is None:
        return None
    # repack to pass the type check
    result = []
    for ele in output_symbols:
        assert ele is not None
        result.append(ele)
    return result


def match_sole_return_value(node) -> Union[VarBase, None]:
    output_symbols = _match_return_values(node, 1)
    if output_symbols is None:
        return None
    return output_symbols[0]


# TODO:  generalize this function to match_edge_linear
# TODO: distinguish matmul, inner_product, outer_product by shape inference
def match_node_linear(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
) -> Union[list[OpBase], None]:
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
        assert isinstance(rhs_node.func, ast.Name)
        if rhs_node.func.id != "linear":
            return None
        if len(rhs_node.args) != 2:
            return None
        if (
            isinstance(rhs_node.args[0], ast.Attribute)
            and isinstance(rhs_node.args[0].value, ast.Name)
            and rhs_node.args[0].value.id == "n"
        ):
            input_symbol = match_vertex_input(rhs_node.args[0])
            weight_symbol = match_weight_var(rhs_node.args[1])
            assert input_symbol is not None and weight_symbol is not None
            return [
                NodeDenseOp._make(
                    {
                        "input": input_symbol,
                        "weight": weight_symbol,
                        "result": output_symbol,
                    }
                )
            ]
        elif (
            isinstance(rhs_node.args[1], ast.Attribute)
            and isinstance(rhs_node.args[1].value, ast.Name)
            and rhs_node.args[1].value.id == "n"
        ):
            input_symbol = match_vertex_input(rhs_node.args[1])
            weight_symbol = match_weight_var(rhs_node.args[0])
            assert input_symbol is not None and weight_symbol is not None
            return [
                NodeDenseOp._make(
                    {
                        "input": input_symbol,
                        "weight": weight_symbol,
                        "result": output_symbol,
                    }
                )
            ]
        else:
            return None
    elif isinstance(rhs_node, ast.BinOp):
        if not isinstance(rhs_node.op, ast.Mult):
            return None
        # if rhs_node.left.value.id == "n":
        if (
            isinstance(rhs_node.left, ast.Attribute)
            and isinstance(rhs_node.left.value, ast.Name)
            and rhs_node.left.value.id == "n"
        ):
            input_symbol = match_vertex_input(rhs_node.left)
            weight_symbol = match_weight_var(rhs_node.right)
            assert input_symbol is not None and weight_symbol is not None
            return [
                NodeDenseOp._make(
                    {
                        "input": input_symbol,
                        "weight": weight_symbol,
                        "result": output_symbol,
                    }
                )
            ]

        elif (
            isinstance(rhs_node.right, ast.Attribute)
            and isinstance(rhs_node.right.value, ast.Name)
            and rhs_node.right.value.id == "n"
        ):
            input_symbol = match_vertex_input(rhs_node.right)
            weight_symbol = match_weight_var(rhs_node.left)
            assert input_symbol is not None and weight_symbol is not None
            return [
                NodeDenseOp._make(
                    {
                        "input": input_symbol,
                        "weight": weight_symbol,
                        "result": output_symbol,
                    }
                )
            ]
        else:
            return None
    return None


def match_unary_functions(
    output_symbols: list[VarBase], loop_type: list[tuple[str, str, str]], rhs_node
) -> Union[list[OpBase], None]:
    """this function matches transpose, concatenation and split"""
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
        assert len(rhs_node.args) == 1
        assert isinstance(rhs_node.func, ast.Name)
        assert rhs_node.func.id == "split"
        input_symbol = match_data_input(rhs_node.args[0])
        if input_symbol is None:
            return None
        return [SplitOp._make({"input": input_symbol, "results": output_symbols})]
    else:
        # output_symbol = match_sole_return_value(node)
        output_symbol = output_symbols[0]
        if output_symbol is None:
            return None
        if isinstance(rhs_node, ast.Call):
            assert isinstance(rhs_node.func, ast.Name)
            if rhs_node.func.id == "transpose":
                input_symbol = match_data_input(rhs_node.args[0])
                if input_symbol is None:
                    return None
                return [
                    TransposeOp._make({"result": output_symbol, "input": input_symbol})
                ]
            elif rhs_node.func.id == "concatenate":
                assert isinstance(rhs_node.args[0], ast.List)
                assert isinstance(rhs_node.args[0].elts[0], ast.Attribute)
                assert isinstance(rhs_node.args[0].elts[0].value, ast.Name)
                print(rhs_node.args[0].elts[0].value.id, rhs_node.args[0].elts[0].attr)
                assert isinstance(rhs_node.args[0].elts[1], ast.Attribute)
                assert isinstance(rhs_node.args[0].elts[1].value, ast.Name)
                print(rhs_node.args[0].elts[1].value.id, rhs_node.args[0].elts[1].attr)
                input_symbol0 = match_data_input(rhs_node.args[0].elts[0])
                input_symbol1 = match_data_input(rhs_node.args[0].elts[0])
                if input_symbol0 is None or input_symbol1 is None:
                    return None
                return [
                    ConcatenateOp._make(
                        {
                            "result": output_symbol,
                            "input": [input_symbol0, input_symbol1],
                        }
                    )
                ]
    return None


# This function matches copy, negation
def match_copy_and_negation(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
) -> Union[list[OpBase], None]:
    # if not isinstance(node, ast.Assign):
    #    return False
    # output_symbol = match_sole_return_value(node)
    # if output_symbol is None:
    #    return None
    # UnaryOp(op=USub(), operand=
    if isinstance(rhs_node, ast.UnaryOp):
        # This is a Negation Op
        if isinstance(rhs_node.op, ast.USub):
            input_symbol = match_data_input(rhs_node.operand)
            if input_symbol is None:
                return None
            return [NegativeOp._make({"input": input_symbol, "result": output_symbol})]
    else:
        input_symbol = match_data_input(rhs_node)
        if input_symbol is None:
            return None
        print(input_symbol, output_symbol)
        return [CopyOp._make({"input": input_symbol, "result": output_symbol})]


def match_nonlinear(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
) -> Union[list[OpBase], None]:
    # if not isinstance(node, ast.Assign):
    #    return False
    # output_symbol = match_sole_return_value(node)
    # if output_symbol is None:
    #    return None
    if not isinstance(rhs_node, ast.Call):
        return None
    assert isinstance(rhs_node.func, ast.Name)
    if not rhs_node.func.id in ["exp", "tanh"]:
        return None
    # todo: incorporate single argument function, i.e., concatenate, here
    if len(rhs_node.args) != 1:
        return None
    print("func name, ", rhs_node.func.id)
    print("input, ", rhs_node.args[0])
    input_symbol = match_data_input(rhs_node.args[0])
    if input_symbol is None:
        return None
    if rhs_node.func.id == "Exponential":
        op_cls = ExponentialOp
    elif rhs_node.func.id == "InverseExponential":
        op_cls = InverseExponentialOp
    elif rhs_node.func.id == "Tanh":
        op_cls = TanhOp
    elif rhs_node.func.id == "InverseTanh":
        op_cls = InverseTanhOp
    elif rhs_node.func.id == "LeakyRelu":
        op_cls = LeakyReluOp
    elif rhs_node.func.id == "InverseLeakyRelu":
        op_cls = InverseLeakyReluOp
    else:
        return None
    return [op_cls._make({"input": input_symbol, "result": output_symbol})]


if __name__ == "__main__":
    print(ast.dump(ast.parse('e["zizj"] = 0.0')))
    print(ast.dump(ast.parse('e["zizj"] = concat([e.zi, e.zj])')))
    print(ast.dump(ast.parse('n["hs"] = Linear(V[n.ntype], n.feature)')))
    print(ast.dump(ast.parse('n["hs"] = -V[n.ntype] * n.feature')))
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
