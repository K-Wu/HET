#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from .def_use_analyzer import DefUseAnalyzerPass
from ...ir.InterOpSSA.utils import OrderedSetQueue
from .base_op_sequencer import BaseOpSequencer


class ShapeInfererPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "ShapeInfererPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [BaseOpSequencer.get_name(), DefUseAnalyzerPass.get_name()]

    def run(self, program: Program) -> list[str]:
        """
        After pattern match from the inter-op IR (or, after differentiation for the backward propagation), we get all base_ops and unique variable names. We need to infer the shapes of all variables.
        Prerequisite: def_use_analyzer.
        """
        worklist = OrderedSetQueue()
        for op in program.base_ops:
            worklist.put(op)
        while not worklist.empty():
            op = worklist.get()
            curr_opr_shape_info = [
                program.var_table.get_shape_info(opr)
                for opr in op.get_operands()
            ]
            curr_res_shape_info = [
                program.var_table.get_shape_info(res)
                for res in op.get_results()
            ]

            opr_shape_info, res_shape_info = op.infer_shape(
                curr_opr_shape_info, curr_res_shape_info
            )
            for opr, shape_info in zip(op.get_operands(), opr_shape_info):
                program.var_table.set_shape_info_or_throw(opr, shape_info)
            for res, shape_info in zip(op.get_results(), res_shape_info):
                program.var_table.set_shape_info_or_throw(res, shape_info)

            for opr, old_shape_info, new_shape_info in zip(
                op.get_results(), curr_res_shape_info, res_shape_info
            ):
                # If an result shape has changed from None to known, move the using ops to the head of the queue
                if old_shape_info is None and new_shape_info is not None:
                    for using_op in program.get_using_ops(opr):
                        worklist.move_to_head(using_op)

            for res, old_shape_info, new_shape_info in zip(
                op.get_operands(), curr_opr_shape_info, opr_shape_info
            ):
                # If an operand shape has changed from known to None, move the defining ops to the head of the queue
                if old_shape_info is not None and new_shape_info is None:
                    defining_op = program.get_defining_op(res)
                    if defining_op is not None:
                        worklist.move_to_head(defining_op)

        return []


# TODO: propagate row_purpose after ShapeInfererPass
# TODO: optionally compactize the shape after row_purpose propagation
