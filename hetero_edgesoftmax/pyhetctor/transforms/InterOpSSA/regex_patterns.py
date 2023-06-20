#!/usr/bin/env python3
import re


def strip_group_names(pattern):
    """Strip group names from pattern string"""
    return re.sub(r"\?P<.*?>", "", pattern)


# Non-weight variable pattern: (EDGEWISE, "varname") or (NODEWISE, "varname")
# or (DSTNODE, "varname") or (SRCNODE, "varname")
non_weight_pattern = (
    r"\((?P<type>EDGEWISE|NODEWISE|DSTNODE|SRCNODE),\"(?P<var_name>[A-z0-9_]*)\"\)"
)

# Weight pattern: (WeightName_0123, SLICETYPE)
# NB: to conform with this simple parser, weight without slicing is denoted as
# (Wname, NONE) rather than mere Wname
weight_pattern = r"\((?P<weight_name>[A-z0-9_]*),(?P<type_slice>[A-Z]*)\)"

keyword_value_pair_pattern = (
    r"(?P<keyword>[a-z0-9_]*)=(?P<value>{weight_pattern}|{non_weight_pattern})".format(
        weight_pattern=strip_group_names(weight_pattern),
        non_weight_pattern=strip_group_names(non_weight_pattern),
    )
)

# operator_pattern = r"(?P<funcname>[A-Z][A-z0-9_]*)\s*\(((?P<keywords>[a-z0-9_]*\s*=\s*.*)\s*,)*\s*(?P<lastkeyword>[a-z0-9_]*\s*=\s*.*\)),?\)\s*;"
# NB: plain assignment e.g. (NODEWISE, "msg") <- (EDGEWISE, "zi") is denoted as
# (NODEWISE, "msg")=Copy(input=(EDGEWISE, "zi"));
operator_pattern = r"((?P<result2>{weight_pattern}|{non_weight_pattern}),)?(?P<result>{weight_pattern}|{non_weight_pattern})=(?P<funcname>[A-Z][A-z0-9_]*)\((?P<keyword_fields>({keyword_value_pair},)*{keyword_value_pair},?)\);".format(
    weight_pattern=strip_group_names(weight_pattern),
    non_weight_pattern=strip_group_names(non_weight_pattern),
    keyword_value_pair=strip_group_names(keyword_value_pair_pattern),
)


# Usage
# For every line, do the following steps:
# 1. strip comments, and whitespaces
# 2. match operator_pattern
# 3. if matched, extract the substring in the match group "keyword_fields", and apply match_all using keyword_value_pair_pattern


# The main routine is provided to aid development process
if __name__ == "__main__":
    print(keyword_value_pair_pattern)
    print(operator_pattern)
