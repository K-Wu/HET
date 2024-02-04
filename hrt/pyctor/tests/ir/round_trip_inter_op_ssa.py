#!/usr/bin/env python3
"""
use round trip to test the serialize/deserialize of inter-op SSA work as intended.
"""
from ...ir.InterOpSSA import variable_tables

filenames = [
    "pyctor/examples/inter-op-ssa/hgt.inter-op-ssa",
    "pyctor/examples/inter-op-ssa/rgcn.inter-op-ssa",
    "pyctor/examples/inter-op-ssa/rgat.inter-op-ssa",
    "pyctor/examples/inter-op-ssa/hgt.bck.inter-op-ssa",
    "pyctor/examples/inter-op-ssa/rgcn.bck.inter-op-ssa",
    "pyctor/examples/inter-op-ssa/rgat.bck.inter-op-ssa",
]


def remove_comment_and_whitespace(lines: list[str]) -> str:
    lines_after_filter = []
    for line in lines:
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = variable_tables.remove_white_spaces(line.strip())
        # skip empty or comment lines
        if len(line) == 0:
            continue
        lines_after_filter.append(line)
    return "\n".join(lines_after_filter)


if __name__ == "__main__":
    for filename in filenames:
        with open(filename) as fd:
            prog = programs.Program.loads(fd.readlines())

            actual_str = prog.dumps()
        with open(filename) as fd:
            expected_str = remove_comment_and_whitespace(fd.readlines())
        assert actual_str == expected_str
