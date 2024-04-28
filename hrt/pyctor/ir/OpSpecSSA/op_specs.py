import abc
from recordclass import dataobject
from ..InterOpSSA.variables import WeightVar
from .var_specs import DataVarSpec, parse_var_spec_class
import json
from typing import Any, Union


class OpSpecBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_opspec_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_opspec_dict(cls, opspec_dict: dict[str, Any]) -> "OpSpecBase":
        raise NotImplementedError


class FinalOpSpecMeta(type(dataobject), type(OpSpecBase)):
    pass


class _TraversalOpSpec(dataobject):
    results: list[Union[DataVarSpec, WeightVar]]
    operands: list[Union[DataVarSpec, WeightVar]]
    op_idx: int
    schedule: dict[str, Any]
    access: dict[str, Any]


class TraversalOpSpec(_TraversalOpSpec, OpSpecBase, metaclass=FinalOpSpecMeta):
    def get_opspec_dict(self) -> dict[str, Any]:
        opspec_dict: dict[str, Any] = {
            "result": [ele.to_list() for ele in self.results],
            "operands": [ele.to_list() for ele in self.operands],
            "schedule": self.schedule,
            "access": self.access,
        }
        return opspec_dict

    @classmethod
    def from_opspec_dict(
        cls, opspec_dict: dict[str, Any]
    ) -> "TraversalOpSpec":
        operands = [
            parse_var_spec_class(ele["name"]).from_list(ele)
            for ele in opspec_dict["operands"]
        ]
        results = [
            parse_var_spec_class(ele["name"]).from_list(ele)
            for ele in opspec_dict["results"]
        ]
        return cls(
            operands=operands,
            results=results,
            schedule=opspec_dict["schedule"],
            access=opspec_dict["access"],
        )

    def to_string(self) -> str:
        return (
            f"traversal_{self.op_idx}"
            + "{\n"
            + json.dumps(self.get_opspec_dict())
            + "\n}"
        )


class _GEMMOpSpec(dataobject):
    product: Union[DataVarSpec, WeightVar]
    left: Union[DataVarSpec, WeightVar]
    right: Union[DataVarSpec, WeightVar]
    op_idx: int
    edgewise_use_compaction: dict[Union[DataVarSpec, WeightVar], bool]
    schedule: dict[str, Any]
    access: dict[str, Any]


class GEMMOpSpec(_GEMMOpSpec, OpSpecBase, metaclass=FinalOpSpecMeta):
    def get_opspec_dict(self) -> dict[str, Any]:
        opspec_dict: dict[str, Any] = {
            "left": list(self.left.to_list()),
            "right": list(self.right.to_list()),
            "product": list(self.product.to_list()),
            "schedule": self.schedule,
            "access": self.access,
        }
        return opspec_dict

    @classmethod
    def from_opspec_dict(cls, opspec_dict: dict[str, Any]) -> "GEMMOpSpec":
        left = parse_var_spec_class(opspec_dict["left"]["name"]).from_list(
            opspec_dict["left"]
        )
        right = parse_var_spec_class(opspec_dict["right"]["name"]).from_list(
            opspec_dict["right"]
        )
        product = parse_var_spec_class(
            opspec_dict["product"]["name"]
        ).from_list(opspec_dict["product"])
        return cls(
            left=left,
            right=right,
            product=product,
            schedule=opspec_dict["schedule"],
            access=opspec_dict["access"],
        )

    def to_string(self) -> str:
        return (
            f"gemm_{self.op_idx}"
            + "{\n"
            + json.dumps(self.get_opspec_dict())
            + "\n}"
        )
