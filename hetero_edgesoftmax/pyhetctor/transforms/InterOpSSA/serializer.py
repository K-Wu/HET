#!/usr/bin/env python3
from .regex_match import *

# import .regex_match


def strip_white_spaces(line):
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


# Usage
# For every line, do the following steps:
# 1. strip comments, and whitespaces
# 2. match operator_pattern
# 3. if matched, extract the substring in the match group "keyword_fields", and apply match_all using keyword_value_pair_pattern
def loads(lines):
    for line in lines:
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = line.strip()
        line = strip_white_spaces(line)
        # skip empty or comment lines
        if len(line) == 0:
            continue
        match = re.match(operator_pattern, line)
        if match:
            result = match.group("result")
            func_name = match.group("funcname")
            keyword_fields = match.group("keyword_fields")
            keyword_value_pairs = re.findall(keyword_value_pair_pattern, keyword_fields)
            print(result, func_name, keyword_fields)
            # print every matched key_value pair
            for keyword_value_pair in keyword_value_pairs:
                keyword = keyword_value_pair[0]
                value = keyword_value_pair[1]
                print("   ", keyword, value)
        # print(line)


# a simple test
if __name__ == "__main__":
    with open("pyhetctor/examples/inter-op-ssa/hgt.inter-op-ssa") as fd:
        loads(fd)
