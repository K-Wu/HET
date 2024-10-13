from ..ir.InterOpSSA.program_serializer import find_first_level_scopes
from ..ir.InterOpSSA.variable_tables import VariableTable
import json

filenames = [
    "pyctor/examples/op-spec-ssa/edgewise_fused_more.op-spec-ssa",
    "pyctor/examples/op-spec-ssa/edgewise_fused.op-spec-ssa",
    "pyctor/examples/op-spec-ssa/edgewise_unfused.op-spec-ssa",
    "pyctor/examples/op-spec-ssa/edgewise_unfused.bck.op-spec-ssa",
    "pyctor/examples/op-spec-ssa/single_dense.compact.op-spec-ssa",
    "pyctor/examples/op-spec-ssa/single_dense.op-spec-ssa",
]


def check_opspec_valid_json():
    """This function checks if the OPSPEC scope contains a valid json string."""
    for filename in filenames:
        with open(filename) as fd:
            print(f"Processing {filename}")
            lines = fd.readlines()
            scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
            for scope_beg, scope_end, scope_tag in scopes:
                if scope_tag != "OPSPEC":
                    continue
                opspec_scopes: list[
                    tuple[int, int, str]
                ] = find_first_level_scopes(lines[scope_beg + 1 : scope_end])
                for (
                    opspec_scope_beg,
                    opspec_scope_end,
                    opspec_scope_tag,
                ) in opspec_scopes:
                    opspec = json.loads(
                        "".join(
                            "\n".join(
                                lines[
                                    scope_beg
                                    + 1
                                    + opspec_scope_beg
                                    + 1 : scope_beg
                                    + 1
                                    + opspec_scope_end
                                ]
                            )
                        )
                    )
                    print(opspec_scope_tag, opspec)


def check_valid_variable_table():
    """This function checks if the OPSPEC scope contains a valid json string."""
    for filename in filenames:
        with open(filename) as fd:
            print(f"Processing {filename}")
            lines = fd.readlines()
            scopes: list[tuple[int, int, str]] = find_first_level_scopes(lines)
            for scope_beg, scope_end, scope_tag in scopes:
                if scope_tag != "VARIABLETABLE":
                    continue
                var_table = VariableTable.loads(
                    lines[scope_beg : scope_end + 1]
                )
                print(var_table.dumps())


if __name__ == "__main__":
    check_opspec_valid_json()
    check_valid_variable_table()
