import abc
from recordclass import dataobject
from ..InterOpSSA.variables import VarBase
import json


class OpSpecBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_opspec_dict(self) -> dict:
        raise NotImplementedError


class FinalOpSpecMeta(type(dataobject), type(OpSpecBase)):
    pass


class TraversalOpSpec(OpSpecBase, metaclass=FinalOpSpecMeta):
    results: list[VarBase]
    operands: list[VarBase]
    op_idx: int

    def __init__(self):
        ...

    def get_opspec_dict(self) -> dict:
        raise NotImplementedError

    def to_string(self) -> str:
        return (
            f"traversal_{self.op_idx}"
            + "{\n"
            + json.dumps(self.get_opspec_dict())
            + "\n}"
        )


class GEMMOpSpec(OpSpecBase, metaclass=FinalOpSpecMeta):
    results: list[VarBase]
    operands: list[VarBase]

    def __init__(self):
        ...

    def get_opspec_dict(self) -> dict:
        raise NotImplementedError

    def to_string(self) -> str:
        return (
            f"gemm_{self.op_idx}"
            + "{\n"
            + json.dumps(self.get_opspec_dict())
            + "\n}"
        )
