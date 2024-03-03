#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from ...ir.InterOpSSA.operators import OpBase
from ...ir.InterOpSSA.variable_tables import DefUseEntry
from .value_numberer import ValueNumbererPass


class DefUseAnalyzerPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "DefUseAnalyzerPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        # As a prerequisite, we must get the list input variables and weight variables before this pass.
        return [ValueNumbererPass.get_name()]

    def run(self, program: Program) -> list[str]:
        """
        This method does def-use chain analysis on all the operations in a
        program, and creates def_use_table.
        No fused op is allowed.
        Prerequisite: value_numberer (for simplicity, let's just run the full value_numbering for now)
        Invalidate: None
        """
        ops: list[OpBase] = list(program.yield_base_ops())
        if len(self.def_use_table) != 0:
            print("Warning: def_use_table is not empty, will be overwritten.")
        # Reset the def_use_table to an empty dict
        self.def_use_table = dict()

        # Step 1 create dummy entry for input variables and weight variables
        for var in program.var_table.vars_input:
            # Set def_op as None to indicate input and weight variables
            entry = DefUseEntry(
                name=program.var_table.get_var_key_str(
                    var, after_value_numbering=True
                ),
                def_op=None,
                use_ops=[],
            )
            self.def_use_table[
                program.var_table.get_var_key_str(
                    var, after_value_numbering=True
                )
            ] = [entry]

        # Step 2 process every operation
        for op in ops:
            # Each definition corresponds to one DefUseEntry.
            # Before ssa numbering is done, key value pair in the dict is (var_key, list[DefUseEntry])
            # After ssa numbering is done, key value pair in the dict is (value (var_namen), DefUseEntry)
            for res in {*op.get_results()}:
                entry = DefUseEntry(name=res.get_name(), def_op=op, use_ops=[])
                # Whether after_value_numbering is True or not, we don't need to (calculate and ) refer to numbered_val_to_key to find the dictionary key
                if res not in self.def_use_table:
                    self.def_use_table[
                        program.var_table.get_var_key_str(
                            res, after_value_numbering=True
                        )
                    ] = []
                dict_record: list[DefUseEntry] = self.def_use_table[
                    program.var_table.get_var_key_str(
                        res, after_value_numbering=True
                    )
                ]
                assert isinstance(dict_record, list)
                dict_record.append(entry)

            for opr in {*op.get_operands()}:
                # Whether after_value_numbering is True or not, self.get_var_key_str returns the variable name (i.e., the name before _numberedXX) as the dictionary key
                assert opr in self.def_use_table
                dict_record = self.def_use_table[
                    program.var_table.get_var_key_str(
                        opr, after_value_numbering=True
                    )
                ]
                assert isinstance(dict_record, list)
                dict_record[-1].use_ops.append(op)

        return []
