#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from ...ir.InterOpSSA.operators import OpBase, FusedOpBase
from typing import (
    Union,
    Generator,
)


def calc_base_op_to_base_seq(
    operations: list[Union[OpBase, FusedOpBase]]
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


class BaseOpSequencer(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "BaseOpSequencer"

    def get_prerequisites(self, program: Program) -> list[str]:
        return []

    def run(self, program: Program) -> list[str]:
        """ """
        program.base_op_to_base_seq_no = calc_base_op_to_base_seq(
            program.base_ops
        )

        return []
