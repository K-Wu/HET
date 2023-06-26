#!/usr/bin/env python3
from . import regex_patterns
import re
from .context import strip_white_spaces, VariableTable
from typing import Tuple, Union


def find_scope_end(lines: list[str], scope_beg: int) -> int:
    """Find the end of a scope, given the beginning of the scope"""
    scopes_in_between = 0
    for idx_line, line in enumerate(lines[scope_beg + 1 :]):
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


def loads_op(lines: list[str]) -> list[dict["str", Union["str", list[str]]]]:
    # For every line, do the following steps:
    # 1. strip comments, and whitespaces
    # 2. match operator_pattern
    # 3. if matched, extract the substring in the match group "keyword_fields", and apply match_all using keyword_value_pair_pattern
    results = []
    scope_beg = 0
    for idx_line, line in enumerate(lines):
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = line.strip()
        line = strip_white_spaces(line)
        # skip empty or comment lines
        if len(line) == 0:
            continue
        if line.find("{") != -1:
            # beginning of a scope
            scope_beg = idx_line
            scope_tags = line[: line.find("{")].split(",")
            # TODO: handle scope tags
        elif line.find("}") != -1:
            # end of a scope
            scope_end = idx_line
            # assert the line should only contain the scope-end symbol "}" for simplicity of parsing for now
            assert len(line) == 1
            # TODO: handle scope tags
        match = re.match(regex_patterns.operator_pattern, line)
        if match:
            curr_op = dict()
            result = match.group("result")
            func_name = match.group("funcname")
            keyword_fields = match.group("keyword_fields")
            keyword_value_pairs = re.findall(
                regex_patterns.keyword_value_pair_pattern, keyword_fields
            )
            # print(result, func_name, keyword_fields)
            # handle SplitOp
            if match.group("result2"):
                curr_op["results"] = [match.group("result2"), result]
            else:
                curr_op["result"] = result
            curr_op["func_name"] = func_name
            # print every matched key_value pair
            for keyword_value_pair in keyword_value_pairs:
                keyword = keyword_value_pair[0]
                value = keyword_value_pair[1]
                # print("   ", keyword, value)
                # curr_op["keyword_value_pairs"].append((keyword, value))
                curr_op[keyword] = value
            results.append(curr_op)
    return results


def loads_shape_table(lines: list[str]) -> Tuple[VariableTable, int]:
    """Find the scope with ShapeTable tag, and pass the lines in between to VariableTable.loads"""
    shapetable_scope_beg = -1
    for idx_line, line in enumerate(lines):
        if line.find("{") != -1 and line.find("ShapeTable") != -1:
            shapetable_scope_beg = idx_line
            break
    if shapetable_scope_beg == -1:
        raise ValueError("ShapeTable not found")
    shapetable_scope_end = find_scope_end(lines, shapetable_scope_beg)
    return (
        VariableTable.loads(lines[shapetable_scope_beg : shapetable_scope_end + 1]),
        shapetable_scope_end + 1,
    )


def loads(lines: list[str]) -> Tuple[VariableTable, list[dict]]:
    var_table, idx_line = loads_shape_table(lines)
    return var_table, loads_op(lines[idx_line:])


# TODO: implement this
def dumps(operations) -> str:
    """Returns a multi-line string that represents the operations, i.e., in the format of .inter-op-ssa file"""
    raise NotImplementedError


# a simple test
if __name__ == "__main__":
    with open("pyhetctor/examples/inter-op-ssa/hgt.inter-op-ssa") as fd:
        print(loads_op(fd.readlines()))
