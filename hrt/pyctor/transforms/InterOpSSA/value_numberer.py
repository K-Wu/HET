#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from ...ir.InterOpSSA.operators import OpBase
from ...ir.InterOpSSA.variable_tables import VarBase, VariableTable
from .base_op_sequencer import BaseOpSequencer


def register_input_and_weight_var(
    var_table: VariableTable, var: VarBase
) -> None:
    """
    This method is called to register a variable that is the input data or weight variable
    """
    var_table.vars_input.add(var)
    register_value_zero(var_table, var)


def register_value_zero(var_table: VariableTable, var: VarBase) -> None:
    """
    Register the first value of a variable name.
    """
    var_table.numbered_key_vals[var_table.get_var_key_str(var)] = [var]
    var_table.numbered_val_to_key[var] = var


def increase_and_register_value_number(
    var_table: VariableTable, var: VarBase
) -> VarBase:
    """
    This method creates and returns a new variable indicating numbered value
    of var, and add it to the table.
    This method is to be used after the Inter Op SSA program is complete,
    i.e., after the lowering from Inter Op DSL to Inter Op SSA.
    At this time all the base_ops from the statements should be ready for
    analysis. And this process is usually called during the value numbering,
    when def-use chain analysis has been done.
    """
    # For now, we use the _numbered suffix to number values
    new_var = var_table._create_var_by(var, "numbered")
    var_table.numbered_val_to_key[new_var] = var
    var_table.numbered_key_vals[var_table.get_var_key_str(var)].append(new_var)
    return new_var


def _do_value_number_on_program(
    var_table: VariableTable, base_ops: list[OpBase]
) -> tuple["VariableTable", list[OpBase]]:
    """
    This method does value numbering on all the base_ops in a program.
    numbered_key_vals and numbered_val_to_key will be updated accordingly.
    Notice that this function should only be applied on unnumbered program,
    otherwise it will malfunction.
    """
    var_table = VariableTable()
    new_base_ops = []
    for op in base_ops:
        new_op: OpBase = op
        # Use set to deduplicate for cases where one operand/result shows multiple times, so that only one replacement for all these occurrence will be applied
        for opr in {*op.get_operands()}:
            # For operands, use the latest numbered value
            if var_table.get_var_key_str(opr) in var_table.numbered_key_vals:
                new_opr = var_table.numbered_key_vals[
                    var_table.get_var_key_str(opr)
                ][-1]
                new_op = new_op.replace_all_operands_with(opr, new_opr)
            else:
                # register the variable as data input or weights
                register_input_and_weight_var(var_table, opr)

        for opr in {*op.get_results()}:
            # For results, increase the value number if already defined
            if var_table.get_var_key_str(opr) in var_table.numbered_key_vals:
                # increment the number
                new_opr = increase_and_register_value_number(var_table, opr)
                new_op = new_op.replace_all_results_with(opr, new_opr)
            else:
                # register the 0th value of the variable
                register_value_zero(var_table, opr)
        new_base_ops.append(new_op)
    return var_table, new_base_ops


class ValueNumbererPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "ValueNumbererPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [BaseOpSequencer.get_name()]

    def do_data_input_and_weight_var_analysis(
        self, program: Program
    ) -> list[str]:
        """This method does the same thing as `run(self, program: Program) -> list[str]`, formerly named as do_value_number_on_program, except that it does not store the results of numbered_key_vals and numbered_val_to_key. It only stores the results of vars_input."""
        new_var_table, _ = _do_value_number_on_program(
            program.var_table, program.base_ops
        )
        program.var_table.vars_input = new_var_table.vars_input
        return []

    def run(self, program: Program) -> list[str]:
        new_var_table, new_base_ops = _do_value_number_on_program(
            program.var_table, program.base_ops
        )
        program.var_table.numbered_key_vals = new_var_table.numbered_key_vals
        program.var_table.numbered_val_to_key = (
            new_var_table.numbered_val_to_key
        )
        program.var_table.vars_input = new_var_table.vars_input
        program.base_ops = new_base_ops
        # TODO: if changed
        return [BaseOpSequencer.get_name()]
