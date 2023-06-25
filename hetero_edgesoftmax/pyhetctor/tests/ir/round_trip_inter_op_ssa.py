from ...ir.InterOpSSA import op_serializer
from typing import Union
import io

filenames = [
    "pyhetctor/examples/inter-op-ssa/hgt.inter-op-ssa",
    "pyhetctor/examples/inter-op-ssa/rgcn.inter-op-ssa",
    "pyhetctor/examples/inter-op-ssa/rgat.inter-op-ssa",
    "pyhetctor/examples/inter-op-ssa/hgt.bck.inter-op-ssa",
    "pyhetctor/examples/inter-op-ssa/rgcn.bck.inter-op-ssa",
    "pyhetctor/examples/inter-op-ssa/rgat.bck.inter-op-ssa",
]


def remove_comment_and_whitespace(lines: Union[list[str], io.TextIOWrapper]) -> str:
    lines_after_filter = []
    for line in lines:
        if line.find("//") != -1:
            line = line[: line.find("//")]
        line = line.strip()
        line = op_serializer.strip_white_spaces(line)
        # skip empty or comment lines
        if len(line) == 0:
            continue
        lines_after_filter.append(line)
    return "\n".join(lines_after_filter)


if __name__ == "__main__":
    for filename in filenames:
        with open(filename) as fd:
            operations = op_serializer.loads(fd)
            actual_str = op_serializer.dumps(operations)
        with open(filename) as fd:
            expected_str = remove_comment_and_whitespace(fd)
        assert actual_str == expected_str
