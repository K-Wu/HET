#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from ...ir.InterOpSSA.operators import OpBase, UnrealizedBinaryOp, FusedOpBase
from .value_numberer import ValueNumbererPass
from .def_use_analyzer import DefUseAnalyzerPass
from .shape_inferer import ShapeInfererPass
from .base_op_sequencer import BaseOpSequencer


class BinopRealizerPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "BinopRealizerPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [BaseOpSequencer.get_name(), ShapeInfererPass.get_name()]

    def run(self, program: Program) -> list[str]:
        """Realize Unrealized.* ops after shape has been inferred.
        Prerequisite: shape_inferer.
        Invalidate shape_inferer, def_use_analyzer, and value_numberer if changes have been made.
        """
        op_replacement: dict[OpBase, OpBase] = dict()
        for op in program.base_ops:
            if isinstance(op, UnrealizedBinaryOp):
                curr_opr_shape_info = [
                    program.var_table.get_shape_info(opr)
                    for opr in op.get_operands()
                ]
                curr_res_shape_info = [
                    program.var_table.get_shape_info(res)
                    for res in op.get_results()
                ]
                new_op = op.realize(curr_opr_shape_info, curr_res_shape_info)
                op_replacement[op] = new_op

        # Replace the op with the new in program.base_ops
        for idx, op in enumerate(program.base_ops):
            if op in op_replacement:
                program.base_ops[idx] = op_replacement[op]

        # If later used, redo the shape analysis if it is done before because there is new information
        # If later used, redo the def-use chain analysis if it is done before because there is operator changes
        # TODO: if changed
        return [
            # ValueNumbererPass.get_name(),
            BaseOpSequencer.get_name(),
            DefUseAnalyzerPass.get_name(),
            ShapeInfererPass.get_name(),
        ]
