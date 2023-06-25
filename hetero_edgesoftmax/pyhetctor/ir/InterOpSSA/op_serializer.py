#!/usr/bin/env python3
from . import regex_patterns
import re
from typing import Union
import io


def strip_white_spaces(line: str) -> str:
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


# For every line, do the following steps:
# 1. strip comments, and whitespaces
# 2. match operator_pattern
# 3. if matched, extract the substring in the match group "keyword_fields", and apply match_all using keyword_value_pair_pattern
def loads(lines: Union[list[str], io.TextIOWrapper]) -> list[dict]:
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
            curr_op["result"] = result
            curr_op["func_name"] = func_name
            curr_op["keyword_value_pairs"] = []
            # print every matched key_value pair
            for keyword_value_pair in keyword_value_pairs:
                keyword = keyword_value_pair[0]
                value = keyword_value_pair[1]
                # print("   ", keyword, value)
                curr_op["keyword_value_pairs"].append((keyword, value))
            results.append(curr_op)
    return results


# TODO: implement this
# returns a multi-line string that represents the operations, i.e., in the format of .inter-op-ssa file
def dumps(operations) -> str:
    raise NotImplementedError


# a simple test
if __name__ == "__main__":
    with open("pyhetctor/examples/inter-op-ssa/hgt.inter-op-ssa") as fd:
        print(loads(fd))
