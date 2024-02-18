#!/usr/bin/env python3
import ast
from ..InterOpSSA.operators import OpBase
from ..InterOpSSA.operators import *
from ..InterOpSSA.variables import VarBase
from ..InterOpSSA.variables import *
from ..InterOpSSA.programs import VariableTable
from typing import Union


def determine_loop_type(loop_type: list[tuple[str, str, str]]) -> str:
    """
    this function determines the loop type
    """
    types = [item[2] for item in loop_type]
    if types == ["nodes"]:
        return "NODEWISE"
    elif types == ["dst_nodes"]:
        return "DSTNODE"
    elif types == ["dst_nodes", "incoming_edges"]:
        return "EDGEWISE"
    elif types == ["edges"]:
        return "EDGEWISE"
    else:
        raise ValueError("cannot determine the loop type")


def match_loop_nest_and_result(
    var_table: VariableTable, loop_root_node
) -> Union[list[OpBase], None]:
    """
    this is the entry point for pattern matchers. It figures out the for loop type and passes the return variable to the corresponding right-hand-side-only matchers.
    """
    # Note that there might be a chain of operations in one line, e.g.,
    # n.output = n.feature * transpose(W)
    # TO deal with this, we need to do the match in the following steps after canonicalization pass
    # 1. match assignment. This helps us figure out the output
    # 2. recursively match the right hand side of the assignment, where the right hand side entry function will be finally called at the end of any match function, i.e., after all other non-chain match logic failed
    # note that upon entering the right-hand-side matching function, pass in 1) left-hand side results, and 2) for-loop levels. During recursive call, we can pass in the temporary result name instead. For example.
    # when matchiing n.output = n.feature * transpose(W), the first time right-hand-side match is called, "node-wise iteration" and "n.output" are passed in. The second time right-hand-side match is called, "node-wise iteration" and "n.output_tmp1" are passed in

    # Step 1: find the statement node inside loop nest and determine the loop-nest type
    assert isinstance(loop_root_node, ast.For)
    curr_node = loop_root_node
    loop_type: list[tuple[str, str, str]] = []
    # To retrive type, execute type = [item[2] for item in loop_type]
    # types in examples involve ["nodes"], ["dst_nodes"], ["dst_nodes", "incoming_edges"], ["edges"]
    while isinstance(curr_node, ast.For):
        assert isinstance(curr_node.target, ast.Name)
        assert isinstance(curr_node.iter, ast.Call)
        assert isinstance(curr_node.iter.func, ast.Attribute)
        assert isinstance(curr_node.iter.func.value, ast.Name)
        loop_type.append(
            (
                curr_node.target.id,
                curr_node.iter.func.value.id,
                curr_node.iter.func.attr,
            )
        )
        curr_node = curr_node.body[0]

    # Step 2: match the result
    assert isinstance(curr_node, ast.Assign) or isinstance(
        curr_node, ast.AugAssign
    )
    assign_type = (
        "assign" if isinstance(curr_node, ast.Assign) else "augAssign"
    )
    output_symbol = match_sole_return_value(var_table, loop_type, curr_node)
    if output_symbol is None:
        output_symbol = match_dual_return_values(
            var_table, loop_type, curr_node
        )
        if output_symbol is None or (None in output_symbol):
            raise ValueError("cannot recognize the output symbol")
        # match splitOp
        return match_unary_functions_and_throw_unmatched_func_calls(
            var_table, output_symbol, loop_type, curr_node.value
        )

    # Step 3: match the right hand side
    return match_right_hand_side_expr(
        var_table, output_symbol, loop_type, curr_node.value, assign_type
    )


def match_right_hand_side_expr(
    var_table: VariableTable,
    output_symbol: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
    assign_type: str,
) -> Union[list[OpBase], None]:
    """
    This is the entry function to match right hand side expression recursively so that chain of operations could be matched
    """
    # ops = match_nonlinear(output_symbol, loop_type, curr_node.value)
    # if ops:
    #    return ops
    if assign_type == "augAssign":
        # This is a minimal replication of the addition portion in
        # match_binary_op_and_throw_unmatched
        input_symbol1, ops1 = match_data_input(
            output_symbol, var_table, loop_type, rhs_node
        )
        assert input_symbol1 is not None
        return ops1 + [
            UnrealizedAddOp(
                result=output_symbol, left=output_symbol, right=input_symbol1
            )
        ]
    elif assign_type == "assign":
        ops = match_copy_and_negation(
            var_table, output_symbol, loop_type, rhs_node
        )
        if ops:
            return ops
        ops = match_dense_func_call(
            var_table, output_symbol, loop_type, rhs_node
        )
        if ops:
            return ops
        ops = match_outer_product(
            var_table, output_symbol, loop_type, rhs_node
        )
        if ops:
            return ops
        # match_unary_functions_and_throw_unmatched_func_calls should occur after all remaining functions because it will catch any unmatched function calls
        ops = match_unary_functions_and_throw_unmatched_func_calls(
            var_table, [output_symbol], loop_type, rhs_node
        )
        if ops:
            return ops
        ops = match_zero_initialize(output_symbol, loop_type, rhs_node)
        if ops:
            return ops
        ops = match_binary_op_and_throw_unmatched(
            var_table, output_symbol, loop_type, rhs_node
        )
        if ops:
            return ops
    else:
        raise ValueError("unknown assignment type")


def match_binary_op_and_throw_unmatched(
    var_table: VariableTable,
    target: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    """
    this function matches binary operations
    like:
    for n in g.nodes():
        n["hs"] = V[n.ntype] * n.feature
    or
    for e in g.edges():
        e["inner_prod"] = e.src["hs"] * e.dst["hs"]
    """
    if not isinstance(rhs_node, ast.BinOp):
        return None
    if rhs_node.op == ast.Add:
        # Possible operators: ScalarAdd, MatrixAdd, VectorAdd
        # These operators are lowered to UnrealizedAdd for now and handle after shape inference
        input_symbol0, ops0 = match_data_input(
            target, var_table, loop_type, rhs_node.left
        )
        input_symbol1, ops1 = match_data_input(
            target, var_table, loop_type, rhs_node.right
        )
        assert input_symbol0 is not None and input_symbol1 is not None
        return (
            ops0
            + ops1
            + [
                UnrealizedAddOp(
                    result=target, left=input_symbol0, right=input_symbol1
                )
            ]
        )
    elif rhs_node.op == ast.Div:
        # We only support ScalarDivide for now so it will be lowered to ScalarDivide.
        input_symbol0, ops0 = match_data_input(
            target, var_table, loop_type, rhs_node.left
        )
        input_symbol1, ops1 = match_data_input(
            target, var_table, loop_type, rhs_node.right
        )
        assert input_symbol0 is not None and input_symbol1 is not None
        return (
            ops0
            + ops1
            + [
                ScalarDivideOp(
                    result=target, left=input_symbol0, right=input_symbol1
                )
            ]
        )

    elif rhs_node.op == ast.Mult:
        # Possible operators: EdgeInnerProduct, ScalarMultiply, EdgeScalarVectorMul, NodeDense, EdgeDense
        # We determine whether it is a dense layer by checking if one side is a weight variable
        # the rest will be lowered to UnrealizedMul for now and handled after shape inference

        if is_weight_var(rhs_node.right) or is_weight_var(rhs_node.left):
            # This is a NodeDense or EdgeDense
            # NB: op matched. All mismatch will be an error
            if is_weight_var(rhs_node.right):
                assert isinstance(rhs_node.left, ast.Attribute)
                assert isinstance(rhs_node.left.value, ast.Name)
                input_symbol, ops0 = match_data_input(
                    target, var_table, loop_type, rhs_node.left
                )
                weight_symbol, ops1 = match_weight_var(
                    var_table, rhs_node.right
                )
                assert input_symbol is not None and weight_symbol is not None
                if rhs_node.left.value.id == "n":
                    assert (
                        weight_symbol.slice_type == "NODETYPE"
                        or weight_symbol.slice_type == "NONE"
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            NodeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=target,
                            )
                        ]
                    )
                else:
                    assert (
                        weight_symbol.slice_type == "EDGETYPE"
                        or weight_symbol.slice_type == "NONE"
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            EdgeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=target,
                            )
                        ]
                    )
            else:
                assert isinstance(rhs_node.right, ast.Attribute)
                assert isinstance(rhs_node.right.value, ast.Name)

                input_symbol, ops0 = match_data_input(
                    target, var_table, loop_type, rhs_node.right
                )
                weight_symbol, ops1 = match_weight_var(
                    var_table, rhs_node.left
                )
                assert input_symbol is not None and weight_symbol is not None
                if rhs_node.right.value.id == "n":
                    assert (
                        weight_symbol.slice_type == "NODETYPE"
                        or weight_symbol.slice_type == "NONE"
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            NodeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=target,
                            )
                        ]
                    )
                else:
                    assert (
                        weight_symbol.slice_type == "EDGETYPE"
                        or weight_symbol.slice_type == "NONE"
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            EdgeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=target,
                            )
                        ]
                    )
        else:
            # Lower this op to an UnrealizedMul for EdgeInnerProduct, ScalarMultiply, EdgeScalarVectorMul
            input_symbol0, ops0 = match_data_input(
                target, var_table, loop_type, rhs_node.left
            )
            input_symbol1, ops1 = match_data_input(
                target, var_table, loop_type, rhs_node.right
            )
            assert input_symbol0 is not None and input_symbol1 is not None
            return (
                ops0
                + ops1
                + [
                    UnrealizedMulOp(
                        result=target, left=input_symbol0, right=input_symbol1
                    )
                ]
            )

    else:
        raise ValueError("unrecognized binary op")


def is_weight_var(node) -> bool:
    if isinstance(node, ast.Name):
        # sliceless weight
        return True
    if isinstance(node, ast.Subscript):
        # sliced weights, e.g., V[n.ntype]
        if (
            isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Attribute)
            and isinstance(node.slice.value, ast.Name)
        ):
            if node.slice.attr == "ntype" or node.slice.attr == "etype":
                return True
    # Match transpose(WeightVar)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "transpose"
    ):
        return is_weight_var(node.args[0])
    return False


def match_weight_var(
    var_table: VariableTable, node
) -> tuple[Union[WeightVar, None], list[OpBase]]:
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
        return (
            WeightVar.from_dict(
                {"name": node.value.id, "slice_type": slice_type}
            ),
            [],
        )
    elif isinstance(node, ast.Name):
        print("Weight(Sliceless): ", node.id)
        return WeightVar.from_dict({"name": node.id, "slice_type": "NONE"}), []
    # It should be match chain then, we currently only support transpose
    elif isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        assert node.func.id == "transpose"
        assert len(node.args) == 1
        weight_symbol, ops = match_weight_var(var_table, node.args[0])
        assert weight_symbol is not None
        output_symbol = WeightVar.from_dict(
            {
                "name": weight_symbol.name + "_transposed",
                "slice_type": weight_symbol.slice_type,
            }
        )
        ops += [TransposeOp(input=weight_symbol, result=output_symbol)]
        return output_symbol, ops
    raise ValueError("unrecognized weight var")


# TODO: distinguish edgewise (g.edges()), from (n.incoming_edges())
def match_unchained_edge_input(
    var_table: VariableTable, loop_type: list[tuple[str, str, str]], node
) -> Union[DataVar, None]:
    if isinstance(node, ast.Attribute):
        assert isinstance(node.value, ast.Name)
        if node.value.id == "e":
            print("(EDGEWISE) input_key: ", node.attr)
            return DataVar.from_dict({"name": node.attr, "type": "EDGEWISE"})
    return None


# TODO: distinguish edgewise (g.edges()), from (n.incoming_edges())
def _match_edge_output(
    loop_type: list[tuple[str, str, str]], node
) -> Union[DataVar, None]:
    # Distinguish edgewise (g.edges()), from (n.incoming_edges())
    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        assert isinstance(node.value, ast.Name)
        if node.value.id == "e":
            assert isinstance(node.slice, ast.Constant)
            print("(EDGEWISE) output_key: ", node.value.id, node.slice.value)
            return DataVar.from_dict(
                {"name": node.slice.value, "type": "EDGEWISE"}
            )
    # No match chain in output
    return None


def match_zero_initialize(
    output_symbol: VarBase, loop_type: list[tuple[str, str, str]], rhs_node
):
    """lower xxx = 0.0 to an accumulation node or an assertion"""
    if isinstance(rhs_node, ast.Constant):
        if rhs_node.value == 0.0:
            print(output_symbol, "is an accumulation node")
            # TODO: return a hint or assertion
            raise NotImplementedError
    # No match chain in zero_initialize
    # It should be unreachable after this line
    raise ValueError("zero-initialization does not suppoort chaining")


def match_unchained_vertex_input(
    var_table: VariableTable, loop_type: list[tuple[str, str, str]], node
) -> Union[DataVar, None]:
    # distinguish NODEWISE (g.nodes()), from DSTNODE (g.dst_nodes())
    var_type = determine_loop_type(loop_type)
    assert var_type in ["NODEWISE", "DSTNODE"]

    if isinstance(node, ast.Attribute):
        assert isinstance(node.value, ast.Name)
        if node.value.id == "n":
            print("(", var_type, ") input_key: ", node.attr)
            # distinguish DST_NODE, SRC_NODE from NODEWISE
            return DataVar.from_dict({"name": node.attr, "type": var_type})
    return None


def match_data_input(
    output_symbol: VarBase,
    var_table: VariableTable,
    loop_type: list[tuple[str, str, str]],
    node,
) -> tuple[Union[DataVar, None], list[OpBase]]:
    input_symbol = match_unchained_vertex_input(var_table, loop_type, node)
    if input_symbol is None:
        input_symbol = match_unchained_edge_input(var_table, loop_type, node)
    if input_symbol is not None:
        return input_symbol, []

    # It should be match chain after this line. For now, we assume the intermediate data is of the same type, i.e., edgewise or nodewise, as output symbol
    output_temp_symbol = var_table.create_temp_var_dsl(output_symbol)
    # treat chaining as regular assignment, compared to augAssign
    ops = match_right_hand_side_expr(
        var_table, output_temp_symbol, loop_type, node, "assign"
    )
    assert ops is not None
    assert isinstance(output_temp_symbol, DataVar)
    # input_symbol, last_op = var_table.realize_var(output_undef_symbol,ops[-1])
    # ops[-1] = last_op
    return output_temp_symbol, ops


def match_vertex_input(
    output_symbol: VarBase,
    var_table: VariableTable,
    loop_type: list[tuple[str, str, str]],
    node,
) -> tuple[Union[DataVar, None], list[OpBase]]:
    input_symbol, ops = match_data_input(
        output_symbol, var_table, loop_type, node
    )
    # assert vertex output
    if input_symbol is not None:
        assert input_symbol.type in ["NODEWISE", "DSTNODE"]
    return input_symbol, ops


def match_edge_input(
    output_symbol: VarBase,
    var_table: VariableTable,
    loop_type: list[tuple[str, str, str]],
    node,
) -> tuple[Union[DataVar, None], list[OpBase]]:
    input_symbol, ops = match_data_input(
        output_symbol, var_table, loop_type, node
    )
    # assert edge output
    if input_symbol is not None:
        assert input_symbol.type == "EDGEWISE"
    return input_symbol, ops


def _match_vertex_output(
    loop_type: list[tuple[str, str, str]], node
) -> Union[DataVar, None]:
    # Distinguish NODEWISE (g.nodes()), from DSTNODE (g.dst_nodes())
    var_type = determine_loop_type(loop_type)
    assert var_type in ["NODEWISE", "DSTNODE"]

    if isinstance(node, ast.Subscript):
        # output must be sliced, e.g., n["output"]
        assert isinstance(node.value, ast.Name)
        if node.value.id == "n":
            assert isinstance(node.slice, ast.Constant)
            print(
                "(",
                var_type,
                ") output_key: ",
                node.value.id,
                node.slice.value,
            )
            return DataVar.from_dict(
                {"name": node.slice.value, "type": var_type}
            )
    # No match chain in output
    return None


def _match_return_values(
    loop_type: list[tuple[str, str, str]], node, num_return_values
) -> list[Union[VarBase, None]]:
    if len(node.targets) != num_return_values:
        print(node.targets)
        return [None] * num_return_values
    output_symbols = []
    for idx_target in range(len(node.targets)):
        target = node.targets[idx_target]
        output_symbol = _match_vertex_output(loop_type, target)
        if output_symbol is None:
            output_symbol = _match_edge_output(loop_type, target)
        # if mismatch, return None anyway and raise Error in the caller
        output_symbols.append(output_symbol)

    return output_symbols


def match_dual_return_values(
    var_table: VariableTable, loop_type: list[tuple[str, str, str]], node
) -> Union[list[VarBase], None]:
    output_symbols = _match_return_values(loop_type, node, 2)
    if output_symbols[0] is None or output_symbols[1] is None:
        return None
    # repack to pass the type check
    result = []
    for ele in output_symbols:
        assert ele is not None
        result.append(ele)
        var_table.register_dsl_var(ele)
    return result


def match_sole_return_value(
    var_table: VariableTable, loop_type: list[tuple[str, str, str]], node
) -> Union[VarBase, None]:
    output_symbols = _match_return_values(loop_type, node, 1)
    if output_symbols is None or output_symbols[0] is None:
        return None
    var_table.register_dsl_var(output_symbols[0])
    return output_symbols[0]


def match_outer_product(
    var_table: VariableTable,
    output_symbol: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    if not isinstance(rhs_node, ast.Call):
        return None
    assert isinstance(rhs_node.func, ast.Name)
    if not rhs_node.func.id == "outer_product":
        return None
    # NB: op matched. All mismatch will be an error
    # Match-chain handled in the following match function
    input_symbol0, ops0 = match_data_input(
        output_symbol, var_table, loop_type, rhs_node.args[0]
    )
    input_symbol1, ops1 = match_data_input(
        output_symbol, var_table, loop_type, rhs_node.args[1]
    )
    assert input_symbol0 is not None
    assert input_symbol1 is not None
    if determine_loop_type(loop_type) == "EDGEWISE":
        return (
            ops0
            + ops1
            + [
                EdgeOuterProductOp(
                    left=input_symbol0,
                    right=input_symbol1,
                    output=output_symbol,
                )
            ]
        )
    else:
        return (
            ops0
            + ops1
            + [
                NodeOuterProductOp(
                    left=input_symbol0,
                    right=input_symbol1,
                    output=output_symbol,
                )
            ]
        )


def match_dense_func_call(
    var_table: VariableTable,
    output_symbol: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    """
    this function matches dense function operations
    like:
    n["hs"] = linear(V[n.ntype], n.feature)
    Note that operations taking in binop are handled by match_binop, e.g.,
    n["hs"] = V[n.ntype] * n.feature
    """
    if isinstance(rhs_node, ast.Call):
        assert isinstance(rhs_node.func, ast.Name)
        if rhs_node.func.id != "linear":
            return None
        if len(rhs_node.args) != 2:
            return None
        if is_weight_var(rhs_node.args[0]) or is_weight_var(rhs_node.args[1]):
            if is_weight_var(rhs_node.args[1]):
                assert isinstance(rhs_node.args[0], ast.Attribute)
                assert isinstance(rhs_node.args[0].value, ast.Name)
                # NB: op matched. All mismatch will be an error
                weight_symbol, ops1 = match_weight_var(
                    var_table, rhs_node.args[1]
                )
                if rhs_node.args[0].value.id == "n":
                    assert (
                        determine_loop_type(loop_type) == "NODEWISE"
                        or determine_loop_type(loop_type) == "DSTNODE"
                    )
                    input_symbol, ops0 = match_vertex_input(
                        output_symbol, var_table, loop_type, rhs_node.args[0]
                    )
                    assert (
                        input_symbol is not None and weight_symbol is not None
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            NodeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=output_symbol,
                            )
                        ]
                    )
                else:
                    assert determine_loop_type(loop_type) == "EDGEWISE"
                    input_symbol, ops0 = match_edge_input(
                        output_symbol, var_table, loop_type, rhs_node.args[0]
                    )
                    assert (
                        input_symbol is not None and weight_symbol is not None
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            EdgeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=output_symbol,
                            )
                        ]
                    )
            else:
                assert isinstance(rhs_node.args[1], ast.Attribute)
                assert isinstance(rhs_node.args[1].value, ast.Name)
                # NB: op matched. All mismatch will be an error
                weight_symbol, ops1 = match_weight_var(
                    var_table, rhs_node.args[0]
                )
                if rhs_node.args[1].value.id == "n":
                    assert (
                        determine_loop_type(loop_type) == "NODEWISE"
                        or determine_loop_type(loop_type) == "DSTNODE"
                    )
                    input_symbol, ops0 = match_vertex_input(
                        output_symbol, var_table, loop_type, rhs_node.args[1]
                    )
                    assert (
                        input_symbol is not None and weight_symbol is not None
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            NodeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=output_symbol,
                            )
                        ]
                    )
                else:
                    assert determine_loop_type(loop_type) == "EDGEWISE"
                    input_symbol, ops0 = match_edge_input(
                        output_symbol, var_table, loop_type, rhs_node.args[1]
                    )
                    assert (
                        input_symbol is not None and weight_symbol is not None
                    )
                    return (
                        ops0
                        + ops1
                        + [
                            EdgeDenseOp(
                                input=input_symbol,
                                weight=weight_symbol,
                                result=output_symbol,
                            )
                        ]
                    )
        else:
            return None
    return None


def match_unary_functions_and_throw_unmatched_func_calls(
    var_table: VariableTable,
    output_symbols: list[VarBase],
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    """this function matches transpose, concatenation and split, and non-linear functions"""
    if len(output_symbols) != 1:
        # should be split
        # now only SplitOp output multiple variables. This must be a SplitOp
        assert isinstance(rhs_node, ast.Call)
        # NB: op matched. All mismatch will be an error
        assert len(rhs_node.args) == 1
        assert isinstance(rhs_node.func, ast.Name)
        assert rhs_node.func.id == "split"
        # Match-chain handled in the following match function
        input_symbol, ops = match_data_input(
            output_symbols[0], var_table, loop_type, rhs_node.args[0]
        )
        if input_symbol is None:
            return None
        return ops + [SplitOp(input=input_symbol, results=output_symbols)]
    else:
        output_symbol = output_symbols[0]
        if isinstance(rhs_node, ast.Call):
            assert isinstance(rhs_node.func, ast.Name)
            # op matched: must be a unary op
            # NB: op matched. All mismatch will be an error
            if rhs_node.func.id == "transpose":
                # Match-chain handled in the following match function
                input_symbol, ops = match_data_input(
                    output_symbol, var_table, loop_type, rhs_node.args[0]
                )
                if input_symbol is None:
                    raise ValueError("transpose input not found")
                return ops + [
                    TransposeOp(result=output_symbol, input=input_symbol)
                ]
            elif rhs_node.func.id == "concatenate":
                assert isinstance(rhs_node.args[0], ast.List)
                assert isinstance(rhs_node.args[0].elts[0], ast.Attribute)
                assert isinstance(rhs_node.args[0].elts[0].value, ast.Name)
                print(
                    rhs_node.args[0].elts[0].value.id,
                    rhs_node.args[0].elts[0].attr,
                )
                assert isinstance(rhs_node.args[0].elts[1], ast.Attribute)
                assert isinstance(rhs_node.args[0].elts[1].value, ast.Name)
                print(
                    rhs_node.args[0].elts[1].value.id,
                    rhs_node.args[0].elts[1].attr,
                )
                # Match-chain handled in the following match functions
                input_symbol0, ops0 = match_data_input(
                    output_symbol,
                    var_table,
                    loop_type,
                    rhs_node.args[0].elts[0],
                )
                input_symbol1, ops1 = match_data_input(
                    output_symbol,
                    var_table,
                    loop_type,
                    rhs_node.args[0].elts[0],
                )
                if input_symbol0 is None or input_symbol1 is None:
                    raise ValueError("concatenate input not found")
                return (
                    ops0
                    + ops1
                    + [
                        ConcatenateOp(
                            result=output_symbol,
                            input=[input_symbol0, input_symbol1],
                        )
                    ]
                )
            return _match_nonlinear(
                var_table, output_symbol, loop_type, rhs_node
            )
    return None


# This function matches copy, negation
def match_copy_and_negation(
    var_table: VariableTable,
    output_symbol: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    if isinstance(rhs_node, ast.UnaryOp):
        # This is a Negation Op
        if isinstance(rhs_node.op, ast.USub):
            # NB: op matched.
            # Match-chain handled in the following match function
            input_symbol, ops = match_data_input(
                output_symbol, var_table, loop_type, rhs_node.operand
            )
            if input_symbol is None:
                raise ValueError("negation input not found")
            return ops + [NegativeOp(input=input_symbol, result=output_symbol)]
    else:
        # NB: let's do copy evasively here: no chaining is allowed to both avoid complicated pattern match and unnecessary feature support.
        # Match-chain handled in the following match function
        input_symbol, ops = match_data_input(
            output_symbol, var_table, loop_type, rhs_node
        )
        if input_symbol is None:
            raise ValueError("copy input not found")
        print(input_symbol, output_symbol)
        return ops + [CopyOp(input=input_symbol, result=output_symbol)]


def _match_nonlinear(
    var_table: VariableTable,
    output_symbol: VarBase,
    loop_type: list[tuple[str, str, str]],
    rhs_node,
) -> Union[list[OpBase], None]:
    if not isinstance(rhs_node, ast.Call):
        return None
    # NB: op matched. All mismatch will be an error
    assert isinstance(rhs_node.func, ast.Name)
    if not rhs_node.func.id in ["exp", "tanh"]:
        return None
    if len(rhs_node.args) != 1:
        return None
    print("func name, ", rhs_node.func.id)
    print("input, ", rhs_node.args[0])
    input_symbol, ops = match_data_input(
        output_symbol, var_table, loop_type, rhs_node.args[0]
    )
    if input_symbol is None:
        return None
    if rhs_node.func.id == "exp":
        op_cls = ExponentialOp
    elif rhs_node.func.id == "inverse_exp":
        op_cls = InverseExponentialOp
    elif rhs_node.func.id == "tanh":
        op_cls = TanhOp
    elif rhs_node.func.id == "inverse_tanh":
        op_cls = InverseTanhOp
    elif rhs_node.func.id == "leakyrelu":
        op_cls = LeakyReluOp
    elif rhs_node.func.id == "inverse_leakyrelu":
        op_cls = InverseLeakyReluOp
    else:
        raise ValueError("Unknown function: ", rhs_node.func.id)
    return ops + [op_cls(input=input_symbol, result=output_symbol)]


if __name__ == "__main__":
    print(ast.dump(ast.parse('e["zizj"] = 0.0')))
    print(ast.dump(ast.parse('e["zizj"] = concat([e.zi, e.zj])')))
    print(ast.dump(ast.parse('n["hs"] = Linear(V[n.ntype], n.feature)')))
    print(ast.dump(ast.parse('n["hs"] = -V[n.ntype] * n.feature')))
    # print(match_dense_func_call(ast.parse("n[\"hs\"] = Linear(V[n.ntype], n.feature)").body[0]))
    # print(match_dense_func_call(ast.parse("n[\"hs\"] = V[n.ntype] * n.feature").body[0]))
    var_table = VariableTable()
    print(
        match_loop_nest_and_result(
            var_table,
            ast.parse(
                """for n in g.nodes():
        n[\"hs\"] = Linear(V[n.ntype], n.feature)"""
            ).body[0],
        )
    )
    print(
        match_loop_nest_and_result(
            var_table,
            ast.parse(
                """for n in g.nodes():
        n[\"hs\"] = V[n.ntype] * n.feature"""
            ).body[0],
        )
    )
