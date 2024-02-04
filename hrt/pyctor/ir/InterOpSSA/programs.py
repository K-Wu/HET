#!/usr/bin/env python3
from .variables import (
    VarBase,
    is_valid_var_name,
    WeightVar,
)
from .operators import OpBase, FusedOpBase, UnrealizedBinaryOp
from typing import (
    Union,
    Generator,
    Callable,
)

# TODO: Add OpSpecifier to Program
from ..OpSpecSSA.op_specifier import OpSpecifier
from .utils import CallRecord, log_pass_calls, MySet
from .variable_tables import DefUseEntry, VariableTable


class Program:
    passes_call_records: list[
        CallRecord
    ]  # Stores the logging by @log_pass_calls
    passes: MySet[Callable] = MySet()  # Stores analysis and transform passes

    operations: list[Union[OpBase, FusedOpBase]]
    op_to_seq_no: dict[
        OpBase, int
    ]  # fused op is broken down into basic ops in this dict
    var_table: VariableTable

    def calc_op_to_seq(
        self, operations: list[Union[OpBase, FusedOpBase]]
    ) -> dict[OpBase, int]:
        """calculate the operation to sequence id mapping. Fused op will be broken
        down into basic ops and each will be assigned a unique id"""
        op_to_seq_no: dict[OpBase, int] = dict()
        curr_idx = 0
        for op in operations:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    op_to_seq_no[sub_op] = curr_idx
                    curr_idx += 1
            else:
                op_to_seq_no[op] = curr_idx
            curr_idx += 1
        return op_to_seq_no

    def __init__(
        self,
        var_table: VariableTable,
        operations: list[Union[OpBase, FusedOpBase]],
    ):
        self.var_table = var_table
        self.operations = operations
        self.op_to_seq_no = self.calc_op_to_seq(operations)

    # TODO: implement based on get_defining_op
    def get_using_ops(self, var: VarBase) -> list[OpBase]:
        raise NotImplementedError

    def get_defining_op(self, var: VarBase) -> Union[OpBase, None]:
        """returns the operation that defines the variable.
        This function will return None if the variable is an input or weight variable,
        and this function will raise Error if the variable is not found in the program.
        """
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
                return entry.def_op

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

    @passes.register
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

    def yield_basic_ops(self) -> Generator[OpBase, None, None]:
        for op in self.operations:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    yield sub_op
            else:
                yield op

    @passes.register
    @log_pass_calls("realize_ops")
    def realize_ops(self):
        """Realize Unrealized.* ops after shape has been inferred"""

        op_replacement: dict[OpBase, OpBase] = dict()
        for op in self.yield_basic_ops():
            if isinstance(op, UnrealizedBinaryOp):
                curr_opr_shape_info = [
                    self.var_table.get_shape_info(opr)
                    for opr in op.get_operands()
                ]
                curr_res_shape_info = [
                    self.var_table.get_shape_info(res)
                    for res in op.get_results()
                ]
                new_op = op.realize(curr_opr_shape_info, curr_res_shape_info)
                op_replacement[op] = new_op

        # Replace the op with the new in self.operations
        for idx, op in enumerate(self.operations):
            if isinstance(op, FusedOpBase):
                for idx_sub, sub_op in enumerate(op.ops):
                    if op in op_replacement:
                        op.ops[idx_sub] = op_replacement[op]
            else:
                if op in op_replacement:
                    self.operations[idx] = op_replacement[op]

        # Redo the shape analysis if it is done before because there is new information
        done_infer_shape_flag = False
        for pass_record in self.var_table.passes_call_records:
            if self.infer_shapes.__name__ == pass_record.funcname:
                done_infer_shape_flag = True
        if done_infer_shape_flag:
            self.infer_shapes()

        # Redo the def-use chain analysis if it is done before because there is operator changes
        done_value_numbering_flag = False
        done_def_use_chain_analysis = False
        for pass_record in self.var_table.passes_call_records:
            if (
                self.var_table.do_def_use_chain_analysis.__name__
                == pass_record.funcname
            ):
                done_def_use_chain_analysis = True
            if self.var_table.do_value_number_on_program.__name__ == (
                pass_record.funcname
            ):
                done_value_numbering_flag = True
        if done_def_use_chain_analysis:
            self.var_table.do_def_use_chain_analysis(
                list(self.yield_basic_ops()), done_value_numbering_flag
            )

    @passes.register
    @log_pass_calls("infer_shapes")
    def infer_shapes(self):
        """
        after differentiation or pattern match from the inter-op IR, we get all
        operations and unique variable names. We need to infer the shapes of all
        variables.
        """
        for op in self.yield_basic_ops():
            curr_opr_shape_info = [
                self.var_table.get_shape_info(opr) for opr in op.get_operands()
            ]
            curr_res_shape_info = [
                self.var_table.get_shape_info(res) for res in op.get_results()
            ]

            opr_shape_info, res_shape_info = op.infer_shape(
                curr_opr_shape_info, curr_res_shape_info
            )
            for opr, shape_info in zip(op.get_operands(), opr_shape_info):
                self.var_table.set_shape_info_or_throw(opr, shape_info)
            for res, shape_info in zip(op.get_results(), res_shape_info):
                self.var_table.set_shape_info_or_throw(res, shape_info)

        raise NotImplementedError
