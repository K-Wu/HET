#!/usr/bin/env python3
from . import regex_patterns
import re
from .programs import remove_white_spaces, VariableTable, Program
from . import operators
from typing import Union


def find_scope_end(
    lines: list[str],
    scope_beg: int,
    allow_single_line_json_flag: bool = False,
) -> int:
    """Find the end of a scope, given the beginning of the scope.
    This function supports one-line scope, i.e., the scope is in the same line.
    This function does not support { or } other than the beginning and ending of the scope in the strings.
    allow_single_line_json_flag is used in op-spec-ssa serialization/deserialization.
    """
    num_open_scopes = 0
    for idx_line, line in enumerate(lines[scope_beg:]):
        if line.find("{") != -1:
            # Beginning of a scope (which may be the scope whose end is what we are looking for)
            if allow_single_line_json_flag:
                num_open_scopes += len([1 for c in line if c == "{"])
            else:
                assert (
                    sum([1 for c in line if c == "{"]) == 1
                ), "Only one { is allowed in a line"
                num_open_scopes += 1
        if line.find("}") != -1:
            # End of a scope
            if allow_single_line_json_flag:
                num_open_scopes -= len([1 for c in line if c == "}"])
            else:
                assert (
                    sum([1 for c in line if c == "}"]) == 1
                ), "Only one } is allowed in a line"
                num_open_scopes -= 1

            if num_open_scopes == 0:
                # Found the end of the scope
                # Return the line where the end symbol } of the scope is in
                return idx_line + scope_beg
            if num_open_scopes < 0:
                raise ValueError("Unexpected scope end")
    raise ValueError("Scope not closed")


def find_first_level_scopes(lines: list[str]) -> list[tuple[int, int, str]]:
    """Find the first level scopes, and return a dict mapping the scope name to
    the scope beginning and end"""
    scope_beg_end_tags = list()
    idx_line = 0
    while idx_line < len(lines):
        line = lines[idx_line]
        if line.find("{") != -1:
            # This line is the beginning of a scope
            scope_name = line[: line.find("{")]
            scope_beg = idx_line
            # Find the end of the scope and skip the lines in between
            scope_end = find_scope_end(lines, scope_beg)
            # print(scope_beg, scope_end, scope_name)
            scope_beg_end_tags.append((scope_beg, scope_end, scope_name))
            # Assume a new scope should not be on the same line as the previous scope end
            idx_line = scope_end + 1
        else:
            idx_line += 1
    return scope_beg_end_tags


def loads_op(
    lines: list[str],
) -> list[Union[operators.FusedOpBase, operators.OpBase]]:
    # For every line, do the following steps:
    # 1. strip comments, and whitespaces
    # 2. match operator_pattern
    # 3. if matched, extract the substring in the match group "keyword_fields",
    # and apply match_all using keyword_value_pair_pattern
    results = []
    assert remove_white_spaces(lines[0].strip()) == "DAG{"
    assert lines[-1].strip() == "}"
    # Use[1:-1] to avoid adding the outmost DAG{} to the result
    lines = lines[1:-1]
    scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
    curr_scope: Union[None, str] = None
    curr_scope_ops: list[operators.OpBase] = []
    for idx_line, line in enumerate(lines):
        # Strip comments
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = remove_white_spaces(line.strip())
        # Skip empty or comment lines
        if len(line) == 0:
            continue

        # Handle scope tags
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
                    operators.TraversalFusedOp.from_ops(curr_scope_ops)
                )
            else:
                print(curr_scope)
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
                curr_op_data
            )
            if curr_scope is None:
                results.append(curr_op)
            else:
                curr_scope_ops.append(curr_op)
    return results


# Superceded by VariableTable.loads() in hrt/pyctor/ir/InterOpSSA/programs.py
# def loads_shape_table(lines: list[str]) -> tuple[VariableTable, int]:
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


if __name__ == "__main__":
    # Test scope finding
    scopes = find_first_level_scopes(
        ["DAG{", "}", "DAG{ }", "DAG{", "{}", "}"]
    )
    print(scopes)
    print([(0, 1, "DAG"), (2, 2, "DAG"), (3, 5, "DAG")] == scopes)

    # Test program serialization and deserialization
    # The following is essentially the DAG portion in Program.loads() in hrt/pyctor/ir/InterOpSSA/programs.py
    ops = None
    with open("pyctor/examples/inter-op-ssa/hgt.inter-op-ssa") as fd:
        lines = fd.readlines()
        scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
        print("scopes in hgt.inter-op-ssa", scopes)
        for scope_beg, scope_end, scope_tag in scopes:
            # For simplicity of parsing, we assume the scope beginning line only contains tag and "{"
            assert (
                remove_white_spaces(lines[scope_beg].strip())
                == scope_tag + "{"
            )
            # Similarly, we assume the scope ending line only contains "}"
            assert lines[scope_end].strip() == "}"
            if scope_tag.find("DAG") != -1:
                ops = loads_op(lines[scope_beg : scope_end + 1])

                # Set an example to show yaml serialization and deserialization
                # Use .out suffix to avoid git diff
                import yaml

                yaml.dump(ops, open("hgt.inter-op-ssa.temp.yaml.out", "w"))
                yaml.load(
                    open("hgt.inter-op-ssa.temp.yaml.out", "r"),
                    Loader=yaml.Loader,
                )

                # Set an example to show json serialization and deserialization
                import jsonpickle

                jsonpickle.loads(jsonpickle.dumps(ops))
    if ops is None:
        print("DAG not found")
