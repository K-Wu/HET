#!/usr/bin/env python3
from . import regex_patterns
import re
from .variable_tables import remove_white_spaces
from .variables import parse_var_class
from . import operators
from typing import Union
from ...utils.logger import logger, get_oneline_str
from ..OpSpecSSA.op_specs import (
    GEMMOpSpec,
    TraversalOpSpec,
    OpSpecBase,
)


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
                if sum([1 for c in line if c == "{"]) != 1:
                    raise ValueError("Only one { is allowed in a line", line)
                num_open_scopes += 1
        if line.find("}") != -1:
            # End of a scope
            if allow_single_line_json_flag:
                num_open_scopes -= len([1 for c in line if c == "}"])
            else:
                if sum([1 for c in line if c == "}"]) != 1:
                    raise ValueError("Only one } is allowed in a line", line)
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
    the scope beginning and end.
    This method is superceded by the regex-based method. This method now serves as a reference. It has some limitations, e.g., it does not support one-line scope, i.e., the scope is in the same line. Besides, it does not support multiple { or } in a line.
    """
    scope_beg_end_tags = list()
    idx_line = 0
    while idx_line < len(lines):
        line = lines[idx_line]
        if line.find("{") != -1:
            # This line is the beginning of a scope
            scope_name = line[: line.find("{")].strip()  # Remove the indents
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


def find_all_matches(pattern: str, string: str):
    import pcre  # Package https://pypi.org/project/python-pcre/ is required

    """Adapted from https://stackoverflow.com/a/77830224/5555077"""
    pat = pcre.compile(pattern)
    pos = 0
    out = []
    while (match := pat.search(string, pos)) is not None:
        pos = match.span()[1] + 1
        out.append(match)
    return out


# TODO: use dataclass to store the scope information
def find_first_level_scopes_regex(
    lines: list[str],
) -> list[tuple[int, int, str]]:
    multilines = "\n".join(lines)

    # From https://stackoverflow.com/a/35271017/5555077
    # The pattern will match the contents from { to } of each first-level scope.
    pattern = r"\{(?:[^{}]+|(?R))*+\}"

    # Match all and return the results
    matches = find_all_matches(pattern, multilines)

    scope_beg_end_tags = list()
    for match in matches:
        char_beg: int = match.span()[0]
        char_end: int = match.span()[1]
        # Count the number of newlines before the match
        num_newlines_before = multilines.count("\n", 0, char_beg)
        # Count the number of newlines in the match
        num_newlines_in = multilines.count("\n", char_beg, char_end)
        scope_name = multilines[
            multilines[:char_beg].rfind("\n") + 1 : char_beg
        ]
        scope_beg_end_tags.append(
            (
                num_newlines_before,
                num_newlines_before + num_newlines_in,
                scope_name,
            )
        )
    return scope_beg_end_tags


def loads_opspec(lines: list[str]) -> dict[str, OpSpecBase]:
    assert remove_white_spaces(lines[0].strip()) == "OPSPEC{"
    assert lines[-1].strip() == "}"
    # Get the list of opspecs
    op_specs_scope_list = find_first_level_scopes(lines[1:])
    logger.info(get_oneline_str("op_specs_scope_list", op_specs_scope_list))

    results: dict[str, OpSpecBase] = dict()

    # For each opspec of an operator, get the json specificaition
    for (
        op_spec_line_beg,
        op_spec_line_end,
        op_spec_tag,
    ) in op_specs_scope_list:
        import json

        op_spec_line_beg = 1 + op_spec_line_beg
        op_spec_line_end = 1 + op_spec_line_end

        op_spec_dict = json.loads(
            "\n".join(lines[op_spec_line_beg + 1 : op_spec_line_end])
        )
        if "gemm" in op_spec_dict:
            results[op_spec_tag] = GEMMOpSpec.from_opspec_dict(op_spec_dict)
        elif "traversal" in op_spec_dict:
            results[op_spec_tag] = TraversalOpSpec.from_opspec_dict(
                op_spec_dict
            )
        else:
            raise ValueError("Unknown opspec type", op_spec_tag, op_spec_dict)

    return results


def loads_op(
    lines: list[str], test_experimental_find_scope=False
) -> list[Union[operators.FusedOpBase, operators.OpBase]]:
    """This function is used to parse the DAG or DAGDict portion in .inter-op-ssa file.
    It uses regex to match the operators, so whether it is DAGDict, i.e., the operators are values and sequence numbers are keys, or DAG does not matter.
    """
    # For every line, do the following steps:
    # 1. strip comments, and whitespaces
    # 2. match nonfused_operator_pattern
    # 3. if matched, extract the substring in the match group "keyword_fields",
    # and apply match_all using keyword_value_pair_pattern
    results = []
    assert (
        remove_white_spaces(lines[0].strip()) == "DAG{"
        or remove_white_spaces(lines[0].strip()) == "DAGDICT{"
    )
    assert lines[-1].strip() == "}"
    # Use [1:-1] to avoid adding the outmost DAG{} to the result
    lines = lines[1:-1]
    scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
    if test_experimental_find_scope:
        scopes_regex = find_first_level_scopes_regex(lines)
        assert scopes == scopes_regex
    curr_fusion_scope: Union[None, str] = None
    curr_fusion_scope_ops: Union[None, list[operators.OpBase]] = None
    curr_fusion_scope_operands: Union[None, list[operators.VarBase]] = None
    curr_fusion_scope_results: Union[None, list[operators.VarBase]] = None
    for idx_line, line in enumerate(lines):
        # Strip comments
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = remove_white_spaces(line.strip())
        # Skip empty or comment lines
        if len(line) == 0:
            continue

        # Handle this fused operator. The fused operator statement is in the scope tag. The operators in the fused region is in the scope.
        if len(scopes) > 0 and idx_line == scopes[0][0]:
            # Scope opens
            curr_fusion_scope = scopes[0][2]
            curr_fusion_scope_ops = []
            curr_fusion_scope_operands = []
            curr_fusion_scope_results = []
            # Retrieve operands (inputs) and results (outputs) of the fusion scope
            match_operands = re.match(
                regex_patterns.fused_operator_operands_pattern, line
            )
            match_results = re.match(
                regex_patterns.fused_operator_results_pattern, line
            )
            assert match_operands is not None
            assert match_results is not None

            keyword_value_pairs = re.findall(
                regex_patterns.keyword_value_pair_pattern,
                match_operands.group("keyword_fields"),
            )
            for keyword, value in keyword_value_pairs:
                curr_fusion_scope_operands.append(
                    parse_var_class(value).from_string(value)
                )

            results = re.findall(
                regex_patterns.result_pattern, match_results.group("results")
            )
            for result in results:
                curr_fusion_scope_results.append(
                    parse_var_class(value).from_string(result)
                )

            continue
        elif len(scopes) > 0 and idx_line == scopes[0][1]:
            # Scope closes
            # NB:For now we store ops fusion schemes in DAG. We may want to store the fusion scheme separately in future.
            # Put the fused ops into the results

            # Make language server happy
            assert curr_fusion_scope is not None
            assert curr_fusion_scope_results is not None
            assert curr_fusion_scope_operands is not None
            assert curr_fusion_scope_ops is not None

            if curr_fusion_scope.find("GEMMOp") != -1:
                results.append(
                    operators.GEMMFusedOp.from_ops(
                        curr_fusion_scope_results,
                        curr_fusion_scope_operands,
                        curr_fusion_scope_ops,
                    )
                )
            elif curr_fusion_scope.find("TraversalOp") != -1:
                results.append(
                    operators.TraversalFusedOp.from_ops(
                        curr_fusion_scope_results,
                        curr_fusion_scope_operands,
                        curr_fusion_scope_ops,
                    )
                )
            else:
                logger.info(
                    get_oneline_str(
                        "Current fusion scope: ", curr_fusion_scope
                    )
                )
                raise ValueError("unrecognized fused op type!")
            scopes.pop(0)
            curr_fusion_scope = None
            curr_fusion_scope_ops = None
            curr_fusion_scope_operands = None
            curr_fusion_scope_results = None
            continue

        # Match operators
        match = re.match(regex_patterns.nonfused_operator_pattern, line)
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
                # keyword and value are the first two elements of the tuple from the regex match. There are other elements in the tuple, but we don't need them.
                keyword = keyword_value_pair[0]
                value = keyword_value_pair[1]
                # print("   ", keyword, value)
                # curr_op_data["keyword_value_pairs"].append((keyword, value))
                curr_op_data[keyword] = value
            operator_cls = operators.func_name_to_op[func_name]
            curr_op: operators.OpBase = operator_cls.from_keyval_pairs(
                curr_op_data
            )
            if curr_fusion_scope is None:
                results.append(curr_op)
            else:
                assert curr_fusion_scope_ops is not None
                curr_fusion_scope_ops.append(curr_op)
        else:
            logger.warning(
                get_oneline_str("Skipping empty or comment line", line)
            )
    return results


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    # Test scope finding
    scopes = find_first_level_scopes(
        ["DAG{", "}", "DAG{ }", "DAG{", "{}", "}"]
    )
    scopes_regex = find_first_level_scopes_regex(
        ["DAG{", "}", "DAG{ }", "DAG{", "{}", "}"]
    )
    print(scopes)
    print(scopes_regex)
    print([(0, 1, "DAG"), (2, 2, "DAG"), (3, 5, "DAG")] == scopes)

    # Test program serialization and deserialization
    # The following is essentially the DAG portion in Program.loads() in hrt/pyctor/ir/InterOpSSA/programs.py
    TEST_DAG_FILENAMES = [
        # "pyctor/examples/op-spec-ssa/edgewise_fused.op-spec-ssa",
        "pyctor/examples/inter-op-ssa/hgt.inter-op-ssa",
    ]
    for input in TEST_DAG_FILENAMES:
        ops = None
        with open(input) as fd:
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
                    ops = loads_op(
                        lines[scope_beg : scope_end + 1],
                        test_experimental_find_scope=True,
                    )

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

    # Test load op spec
    TEST_OPSPEC_FILENAME = [
        "pyctor/examples/op-spec-ssa/single_dense.compact.op-spec-ssa",
        "pyctor/examples/op-spec-ssa/single_dense.op-spec-ssa",
    ]
    for input in TEST_OPSPEC_FILENAME:
        print("Loading", input)
        ops = None
        with open(input) as fd:
            lines = fd.readlines()
            scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
            print("scopes in hgt.inter-op-ssa", scopes)
            for idx_line_beg, idx_line_end, scope_tag in scopes:
                if scope_tag.find("OPSPEC") != -1:
                    op_specs = loads_opspec(
                        lines[idx_line_beg : idx_line_end + 1]
                    )
                    print(op_specs)
