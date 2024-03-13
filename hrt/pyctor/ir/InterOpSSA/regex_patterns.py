#!/usr/bin/env python3
import re


def strip_group_names(pattern: str) -> str:
    """Strip group names from pattern string"""
    return re.sub(r"\?P<.*?>", "", pattern)


# Non-weight variable pattern: (EDGEWISE, "varname") or (NODEWISE, "varname")
# or (DSTNODE, "varname") or (SRCNODE, "varname")
non_weight_pattern = r"\((?P<type>EDGEWISE|NODEWISE|DSTNODE|SRCNODE),\"(?P<var_name>[A-z0-9_]*)\"\)"

# Weight pattern: (WeightName_0123, SLICETYPE)
# NB: to conform with this simple parser, weight without slicing is denoted as
# (Wname, NONE) rather than mere Wname
weight_pattern = r"\((?P<weight_name>[A-z0-9_]*),(?P<type_slice>[A-Z]*)\)"


result_pattern = r"{weight_pattern}|{non_weight_pattern}".format(
    weight_pattern=strip_group_names(weight_pattern),
    non_weight_pattern=strip_group_names(non_weight_pattern),
)
keyword_value_pair_pattern = (
    r"(?P<keyword>[a-z0-9_]*)=(?P<value>{weight_pattern}|{non_weight_pattern})"
    .format(
        weight_pattern=strip_group_names(weight_pattern),
        non_weight_pattern=strip_group_names(non_weight_pattern),
    )
)

# Non-fused operators have 1--2 results, and fused operators may have more.
# NB: plain assignment e.g. (NODEWISE, "msg") <- (EDGEWISE, "zi") is denoted as
# (NODEWISE, "msg")=Copy(input=(EDGEWISE, "zi"));
nonfused_operator_pattern = r"((?P<result2>{result_pattern}),)?(?P<result>{result_pattern})=(?P<funcname>[A-Z][A-z0-9_]*)\((?P<keyword_fields>({keyword_value_pair},)*{keyword_value_pair},?)\);".format(
    result_pattern=strip_group_names(result_pattern),
    keyword_value_pair=strip_group_names(keyword_value_pair_pattern),
)

fused_operator_results_pattern = r"((?P<results>({result_pattern}),)*({result_pattern}))=(?P<funcname>[A-Z][A-z0-9_]*)".format(
    result_pattern=strip_group_names(result_pattern),
)
fused_operator_operands_pattern = r"=(?P<funcname>[A-Z][A-z0-9_]*)\((?P<keyword_fields>({keyword_value_pair},)*{keyword_value_pair},?)\);".format(
    result_pattern=strip_group_names(result_pattern),
    keyword_value_pair=strip_group_names(keyword_value_pair_pattern),
)


# Usage
# For every line, do the following steps:
# 1. strip comments, and whitespaces
# 2. match nonfused_operator_pattern
# 3. if matched, extract the substring in the match group "keyword_fields", and apply match_all using keyword_value_pair_pattern


# The main routine is provided to aid development process
if __name__ == "__main__":
    print(keyword_value_pair_pattern)
    print(nonfused_operator_pattern)
