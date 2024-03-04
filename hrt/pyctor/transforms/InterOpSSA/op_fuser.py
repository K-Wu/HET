#!/usr/bin/env python3
from ..passes import Pass
from ...ir.InterOpSSA.programs import Program
from .def_use_analyzer import DefUseAnalyzerPass
from .value_numberer import ValueNumbererPass
from .shape_inferer import ShapeInfererPass
from .binop_realizer import BinopRealizerPass
from .base_op_sequencer import BaseOpSequencer
from ...ir.InterOpSSA.utils import OrderedSetQueue


class OpFuserPass(Pass):
    @classmethod
    def get_name(cls) -> str:
        return "OpFuserPass"

    def get_prerequisites(self, program: Program) -> list[str]:
        return [
            BaseOpSequencer.get_name(),
            DefUseAnalyzerPass.get_name(),
            ValueNumbererPass.get_name(),
            ShapeInfererPass.get_name(),
            BinopRealizerPass.get_name(),
        ]

    def run(self, program: Program) -> list[str]:
        """ """

        return []
