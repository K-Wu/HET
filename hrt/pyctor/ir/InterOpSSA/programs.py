#!/usr/bin/env python3
from .variables import (
    VarBase,
    is_valid_var_name,
    WeightVar,
)
from .operators import OpBase, FusedOpBase
from typing import (
    Union,
    Generator,
    Callable,
)

# TODO: Add OpSpecifier to Program
from ..OpSpecSSA.op_specifier import OpSpecifier
from .utils import CallRecord, log_pass_calls
from .variable_tables import DefUseEntry, VariableTable


class Program:
    passes_call_records: list[
        CallRecord
    ]  # Stores the logging by @log_pass_calls

    operations: list[Union[OpBase, FusedOpBase]]
    base_op_to_base_seq_no: dict[
        OpBase, int
    ]  # fused op is broken down into basic ops in this dict
    var_table: VariableTable

    def calc_base_op_to_base_seq(
        self, operations: list[Union[OpBase, FusedOpBase]]
    ) -> dict[OpBase, int]:
        """Calculate the operation to sequence id mapping. Fused op will be broken
        down into basic ops and each will be assigned a unique id.
        This is done at the program instance initialization and operator-fusion invariant.
        """
        base_op_to_base_seq_no: dict[OpBase, int] = dict()
        curr_idx = 0
        for op in operations:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    base_op_to_base_seq_no[sub_op] = curr_idx
                    curr_idx += 1
            else:
                base_op_to_base_seq_no[op] = curr_idx
            curr_idx += 1
        return base_op_to_base_seq_no

    def __init__(
        self,
        var_table: VariableTable,
        operations: list[Union[OpBase, FusedOpBase]],
    ):
        self.var_table = var_table
        self.operations = operations
        self.base_op_to_base_seq_no = self.calc_base_op_to_base_seq(operations)

    def _get_def_use_entry(self, var: VarBase) -> DefUseEntry:
        if not self.var_table.contains(var):
            raise ValueError(
                f"Variable {var} is not found in this program. Make sure the"
                " analysis is run before calling get_defining_op!"
            )
        for entry in self.var_table.def_use_table[
            self.var_table.get_var_key_str(var)
        ]:
            assert isinstance(entry, DefUseEntry)
            if entry.name == var.get_name():
                return entry
        raise ValueError("Variable not found in def-use table!")

    def get_using_ops(self, var: VarBase) -> list[OpBase]:
        return self._get_def_use_entry(var).use_ops

    def get_defining_op(self, var: VarBase) -> Union[OpBase, None]:
        """returns the operation that defines the variable.
        This function will return None if the variable is an input or weight variable,
        and this function will raise Error if the variable is not found in the program.
        """
        # TODO: handle None case, i.e., input or weight variable
        return self._get_def_use_entry(var).def_op

    def get_seqid(self, op: OpBase) -> int:
        """returns the sequence id of the operation"""
        assert op in self.operations
        return self.operations.index(op)

    def assert_define_before_use(self, operand: VarBase, op: OpBase):
        assert self.var_table.contains(operand)
        # operand should either be a weight, or defined before
        if not isinstance(operand, WeightVar):
            operand_def_op = self.get_defining_op(operand)
            if operand_def_op is None:
                # this is an input variable or weight. Skipping the seqid check
                return

            assert self.get_seqid(operand_def_op) < self.get_seqid(op)

    def validate(self) -> None:
        # returns True if 1) every operation has all key-value pairs correctly
        # defined as specified in this file, and 2) use-def chain is correct
        for var in self.var_table.numbered_val_to_key:
            assert self.get_defining_op(var) is not None
            assert is_valid_var_name(var.get_name())
        for op in self.operations:
            op.validate()
            if isinstance(op, FusedOpBase):
                for op_ in op.ops:
                    for operand in op_.get_operands():
                        self.assert_define_before_use(operand, op_)
            else:
                for operand in op.get_operands():
                    self.assert_define_before_use(operand, op)
        return

    def dumps(self) -> str:
        """Returns a multi-line string that contains the variable shape table,
        and operations. In other words, the result is in the format of
        .inter-op-ssa file"""
        result = ""
        result += self.var_table.dumps()
        result += "\nDAG{"
        for op in self.operations:
            result += op.to_string()
            result += "\n"
        result += "}\n"
        return result

    @classmethod
    def loads(cls, lines: list[str]) -> "Program":
        from . import program_serializer

        scopes: list[
            tuple[int, int, str]
        ] = program_serializer.find_first_level_scopes(lines)
        assert len(scopes) == 2
        assert scopes[0][2] == "VariableTable"
        assert scopes[1][2] == "DAG"
        var_table = VariableTable.loads(lines[scopes[0][0] : scopes[0][1] + 1])
        ops = program_serializer.loads_op(lines[scopes[1][0] :])

        return cls(var_table, ops)

    def yield_base_ops(self) -> Generator[OpBase, None, None]:
        for op in self.operations:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    yield sub_op
            else:
                yield op

    # TODO: Move to auto_differer.py
    @log_pass_calls("differentiate")
    def differentiate(self) -> "Program":
        """
        differentiate the program, and return the differentiated program
        """
        diff_ops: list[Union[OpBase, FusedOpBase]] = []
        for op in self.operations:
            diff_ops += op.differentiate()
        # Reconstruct the variable table
        # Notice that if the differentiation is done after forward pass value
        # numbering, the value number chain of the same key may not be preserved
        diff_var_table = self.var_table.differentiate(diff_ops)
        return Program(diff_var_table, diff_ops)
