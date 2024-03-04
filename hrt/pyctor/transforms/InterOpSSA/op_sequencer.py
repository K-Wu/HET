#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from ...ir.InterOpSSA.operators import OpBase, FusedOpBase
from typing import (
    Union,
    Generator,
)


def calc_base_seq_no_to_seq_no(
    operations: list[Union[OpBase, FusedOpBase]]
) -> dict[int, Union[int, tuple[int, int]]]:
    """Calculate the operation to sequence id mapping. Fused op will be broken
    down into basic ops and each will be assigned a unique id.
    This is done at the program instance initialization and operator-fusion invariant.
    """
    base_seq_no_to_seq_no: dict[int, Union[int, tuple[int, int]]] = dict()
    base_seq_no = 0
    seq_no = 0
    for op in operations:
        if isinstance(op, FusedOpBase):
            for idx_sub_op, sub_op in enumerate(op.ops):
                base_seq_no_to_seq_no[base_seq_no] = (seq_no, idx_sub_op)
                base_seq_no += 1
        else:
            base_seq_no_to_seq_no[base_seq_no] = seq_no
            base_seq_no += 1
        seq_no += 1
    return base_seq_no_to_seq_no


class OpSequencer(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "OpSequencer"

    def get_prerequisites(self, program: Program) -> list[str]:
        return []

    def run(self, program: Program) -> list[str]:
        """ """
        program.base_seq_no_to_seq_no = calc_base_seq_no_to_seq_no(
            program.operations
        )

        return []
