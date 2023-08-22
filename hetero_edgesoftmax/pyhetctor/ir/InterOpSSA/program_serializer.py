#!/usr/bin/env python3
from . import regex_patterns
import re
from .programs import strip_white_spaces, VariableTable
from . import operators
from typing import Tuple, Union


def find_scope_end(lines: list[str], scope_beg: int) -> int:
    """Find the end of a scope, given the beginning of the scope"""
    scopes_in_between = 0
    for idx_line, line in enumerate(lines[scope_beg + 1:]):
        if line.find("{") != -1:
            # beginning of another scope
            scopes_in_between += 1
        elif line.find("}") != -1:
            # end of a scope
            scopes_in_between -= 1
            if scopes_in_between == -1:
                # found the end of the scope
                return idx_line + scope_beg + 1
    raise ValueError("Scope not closed")


def find_first_level_scopes(lines: list[str]) -> list[Tuple[int, int, str]]:
    """Find the first level scopes, and return a dict mapping the scope name to
    the scope beginning and end"""
    scope_beg_end_tags = list()
    idx_line = 0
    while idx_line < len(lines):
        line = lines[idx_line]
        if line.find("{") != -1:
            # beginning of a scope
            scope_name = line[: line.find("{")]
            scope_beg = idx_line
            scope_end = find_scope_end(lines, scope_beg)
            scope_beg_end_tags.append((scope_beg, scope_end, scope_name))
            idx_line = scope_end + 1
        else:
            idx_line += 1
    return scope_beg_end_tags


def loads_op(lines: list[str]) -> list[Union[operators.FusedOpBase, operators.OpBase]]:
    # For every line, do the following steps:
    # 1. strip comments, and whitespaces
    # 2. match operator_pattern
    # 3. if matched, extract the substring in the match group "keyword_fields",
    # and apply match_all using keyword_value_pair_pattern
    results = []
    scopes: list[Tuple[int, int, str]] = find_first_level_scopes(lines)
    assert strip_white_spaces(lines[0].strip()) == "DAG{"
    assert strip_white_spaces(lines[-1].strip()) == "}"
    curr_scope: Union[None, str] = None
    curr_scope_ops: list[operators.OpBase] = []
    for idx_line, line in enumerate(lines):
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = line.strip()
        line = strip_white_spaces(line)
        # skip empty or comment lines
        if len(line) == 0:
            continue

        # handle scope tags
        if len(scopes) > 0 and idx_line == scopes[0][0]:
            curr_scope = scopes[0][2]
            continue
        elif len(scopes) > 0 and idx_line == scopes[0][1]:
            # put the fused ops into the results
            assert curr_scope is not None  # make language server happy
            if curr_scope.find("GEMMOp") != -1:
                results.append(operators.GEMMFusedOp.from_ops(curr_scope_ops))
            elif curr_scope.find("TraversalOp") != -1:
                results.append(
                    operators.TraversalFusedOp.from_ops(curr_scope_ops))
            else:
                raise ValueError("unrecognized fused op type!")
            scopes.pop(0)
            curr_scope = None
            continue
        match = re.match(regex_patterns.operator_pattern, line)
        if match:
            curr_op_data: dict["str", Union["str", list[str]]] = dict()
            result = match.group("result")
            func_name = match.group("funcname")
            keyword_fields = match.group("keyword_fields")
            keyword_value_pairs = re.findall(
                regex_patterns.keyword_value_pair_pattern, keyword_fields
            )
            # print(result, func_name, keyword_fields)
            # handle SplitOp
            if match.group("result2"):
                curr_op_data["results"] = [match.group("result2"), result]
            else:
                curr_op_data["result"] = result
            curr_op_data["func_name"] = func_name
            # print every matched key_value pair
            for keyword_value_pair in keyword_value_pairs:
                keyword = keyword_value_pair[0]
                value = keyword_value_pair[1]
                # print("   ", keyword, value)
                # curr_op_data["keyword_value_pairs"].append((keyword, value))
                curr_op_data[keyword] = value
            operator_cls = operators.func_name_to_op[func_name]
            curr_op: operators.OpBase = operator_cls.from_keyval_pairs(
                curr_op_data)
            if curr_scope is None:
                results.append(curr_op)
            else:
                curr_scope_ops.append(curr_op)
    return results


# def loads_shape_table(lines: list[str]) -> Tuple[VariableTable, int]:
#     """Find the scope with ShapeTable tag, and pass the lines in between to
#     VariableTable.loads"""
#     shapetable_scope_beg = -1
#     for idx_line, line in enumerate(lines):
#         if line.find("{") != -1 and line.find("ShapeTable") != -1:
#             shapetable_scope_beg = idx_line
#             break
#     if shapetable_scope_beg == -1:
#         raise ValueError("ShapeTable not found")
#     shapetable_scope_end = find_scope_end(lines, shapetable_scope_beg)
#     return (
#         VariableTable.loads(lines[shapetable_scope_beg : shapetable_scope_end + 1]),
#         shapetable_scope_end + 1,
#     )


def program_loads(
    lines: list[str],
) -> Tuple[VariableTable, list[Union[operators.OpBase, operators.FusedOpBase]]]:
    scopes: list[Tuple[int, int, str]] = find_first_level_scopes(lines)
    var_table = VariableTable.loads(lines[scopes[0][0]: scopes[0][1] + 1])
    ops = loads_op(lines[scopes[1][0]:])
    return var_table, ops


# a simple test
if __name__ == "__main__":
    ops = None
    with open("pyhetctor/examples/inter-op-ssa/hgt.inter-op-ssa") as fd:
        lines = fd.readlines()
        scopes: list[Tuple[int, int, str]] = find_first_level_scopes(lines)
        for scope_beg, scope_end, scope_tag in scopes:
            if scope_tag.find("DAG") != -1:
                ops = loads_op(lines[scope_beg: scope_end + 1])
                import yaml

                # use .out suffix to avoid git diff
                yaml.dump(ops, open("hgt.inter-op-ssa.yaml.out", "w"))
                yaml.load(open("hgt.inter-op-ssa.yaml.out", "r"),
                          Loader=yaml.Loader)
    if ops is None:
        print("DAG not found")
