#!/usr/bin/env python3
from __future__ import annotations
from .variables import VarBase, DataVar, WeightVar, parse_var_class, Shape
from typing import Union, Type, TypeVar  # , Namedtuple
import abc
from recordclass import dataobject
from ...utils.logger import logger


T = TypeVar("T")


class OpBase(metaclass=abc.ABCMeta):
    """when inherited, OpBase provides the interface of operator classes"""

    @classmethod
    def get_opname(cls) -> str:
        return cls.__name__[:-2]  # remove "Op" suffix

    @classmethod
    @abc.abstractmethod
    def from_keyval_pairs(
        cls: Type[T], d: dict["str", Union[list["str"], "str"]]
    ) -> T:
        """
        from_keyval_pairs takes in keyval pairs parsed by op_serializer.py
        recursively rather than instantiated object as keys' values by something
        like namedtuple's cls._make(d), and now we only use namedtuple for
        variables and use recordclass.dataobject for operators
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_keyval_pairs(self) -> dict["str", "str"]:
        """
        to_keyval_pairs returns dict of strings recursively rather than
        instantiated objects as keys' values by something like namedtuple's
        self._asdict(), and now we only use namedtuple for variables and use
        recordclass.dataobject for operators
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_operands(self) -> list[VarBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_results(self) -> list[VarBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        raise NotImplementedError

    @abc.abstractmethod
    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        raise NotImplementedError

    @abc.abstractmethod
    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        raise NotImplementedError


class FinalOpMeta(type(dataobject), type(OpBase)):
    pass


class FusedOpBase(metaclass=abc.ABCMeta):
    """This class is deliberately not inheriting from OpBase to distinguish itself from ordinary ops."""

    results: list[VarBase]
    operands: list[VarBase]
    ops: list[OpBase]

    def __init__(
        self,
        results: list[VarBase],
        operands: list[VarBase],
        ops: list[OpBase],
    ):
        self.results = results
        self.operands = operands
        self.ops = ops

    @classmethod
    def get_opname(cls) -> str:
        raise NotImplementedError

    @classmethod
    def from_ops(
        cls: Type[T],
        results: list[VarBase],
        operands: list[VarBase],
        ops: list[OpBase],
    ) -> T:
        return cls(results, operands, ops)

    def to_string(self) -> str:
        results = ",".join([ele.to_string() for ele in self.results])
        operands = ",".join([ele.to_string() for ele in self.operands])
        result_string = f"{results} = {self.get_opname()}({operands});" + "{\n"
        for op in self.ops:
            result_string += op.to_string()
            result_string += "\n"
        result_string += "}"
        return result_string

    @abc.abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self) -> bool:
        raise NotImplementedError

    def inplace_replace_all_operands_with(
        self: T, old: VarBase, new: VarBase
    ) -> None:
        raise NotImplementedError("Fused ops are not supposed to use this API")

    def inplace_replace_all_results_with(
        self: T, old: VarBase, new: VarBase
    ) -> None:
        raise NotImplementedError("Fused ops are not supposed to use this API")

    def replace_all_operands_with(self: T, old: VarBase, new: VarBase) -> T:
        raise NotImplementedError("Fused ops are not supposed to use this API")

    def replace_all_results_with(self: T, old: VarBase, new: VarBase) -> T:
        raise NotImplementedError("Fused ops are not supposed to use this API")

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        raise NotImplementedError("Fused ops are not supposed to use this API")


class _SplitOp(dataobject):
    results: list[DataVar]
    input: DataVar


class _NodeDenseOp(dataobject):
    result: DataVar
    input: DataVar
    weight: WeightVar


class _WeightDenseOp(dataobject):
    result: WeightVar
    left: WeightVar
    right: WeightVar


class _EdgeDenseOp(dataobject):
    result: DataVar
    input: DataVar
    weight: WeightVar


class _EdgeScalarVectorMulOp(dataobject):
    result: DataVar
    scalar: DataVar
    vector: DataVar


class _UnaryOp(dataobject):
    result: VarBase
    input: VarBase


class _BinaryOp(dataobject):
    result: VarBase
    left: VarBase
    right: VarBase


def replace_if_matched(var: VarBase, old: VarBase, new: VarBase) -> ...:
    if var == old:
        return new
    else:
        return var


#
# Operators that neither fits in ordinary unary ops' fields nor binary ops'
# fields
#
class SplitOp(_SplitOp, OpBase, metaclass=FinalOpMeta):
    # input: DataVar
    # results: list[DataVar]

    def validate(self) -> None:
        for ele in self.get_results():
            ele.validate()

        self.input.validate()

    @classmethod
    def from_keyval_pairs(
        cls: Type["SplitOp"], d: dict["str", Union[list["str"], "str"]]
    ) -> "SplitOp":
        assert d["func_name"] == "Split"
        assert isinstance(d["results"], list)
        results = []
        for ele in d["results"]:
            results.append(DataVar.from_string(ele))
        assert isinstance(d["input"], str)
        input = DataVar.from_string(d["input"])
        return cls(results=results, input=input)

    def to_keyval_pairs(
        self,
    ) -> dict["str", Union["str", list["str"]]]:
        return {
            "results": [ele.to_string() for ele in self.results],
            "func_name": "Split",
            "input": self.input.to_string(),
        }

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def lower(self) -> bool:
        raise NotImplementedError

    def to_string(self) -> str:
        results_str = ",".join([ele.to_string() for ele in self.results])
        return f"{results_str}={self.get_opname()}(input = {self.input})"

    def get_operands(self) -> list[VarBase]:
        return [self.input]

    def get_results(self) -> list[VarBase]:
        # Unpack and pack again to avoid type error
        return [*self.results]

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        # All shapes
        raise NotImplementedError

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.input = replace_if_matched(self.input, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.results = [
            replace_if_matched(ele, old, new) for ele in self.results
        ]

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            results=self.results,
            input=replace_if_matched(self.input, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            results=[
                replace_if_matched(ele, old, new) for ele in self.results
            ],
            input=self.input,
        )


class NodeDenseOp(_NodeDenseOp, OpBase, metaclass=FinalOpMeta):
    # result: DataVar
    # input: DataVar
    # weight: WeightVar

    def validate(self) -> None:
        # Delta of weight is not possible to be the output: it needs outer-product
        assert isinstance(self.result, DataVar)
        assert isinstance(self.input, DataVar)
        assert isinstance(self.weight, WeightVar)
        self.result.validate()
        self.input.validate()
        self.weight.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]) -> "NodeDenseOp":
        result = DataVar.from_string(d["result"])
        input = DataVar.from_string(d["input"])
        weight = WeightVar.from_string(d["weight"])
        return cls(result=result, input=input, weight=weight)

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def lower(self) -> bool:
        raise NotImplementedError

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "input": self.input.to_string(),
            "weight": self.weight.to_string(),
        }

    def to_string(self) -> str:
        return (
            f"{self.result}={self.get_opname()}(input = {self.input}, weight ="
            f" {self.weight})"
        )

    def get_operands(self) -> list[VarBase]:
        return [self.input, self.weight]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.input = replace_if_matched(self.input, old, new)
        self.weight = replace_if_matched(self.weight, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            input=replace_if_matched(self.input, old, new),
            weight=replace_if_matched(self.weight, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=self.input,
            weight=self.weight,
        )

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_matrix_shape()], [
            Shape.get_vector_shape()
        ]


class WeightDenseOp(_WeightDenseOp, OpBase, metaclass=FinalOpMeta):
    # result: WeightVar
    # left: WeightVar
    # right: WeightVar

    def validate(self) -> None:
        # Delta of weight is not possible to be the output: it needs outer-product
        assert isinstance(self.result, WeightVar)
        assert isinstance(self.left, WeightVar)
        assert isinstance(self.right, WeightVar)
        self.result.validate()
        self.left.validate()
        self.right.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]) -> "NodeDenseOp":
        result = WeightVar.from_string(d["result"])
        left = WeightVar.from_string(d["left"])
        right = WeightVar.from_string(d["right"])
        return cls(result=result, left=left, right=right)

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def lower(self) -> bool:
        raise NotImplementedError

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "left": self.left.to_string(),
            "right": self.right.to_string(),
        }

    def to_string(self) -> str:
        return (
            f"{self.result}={self.get_opname()}(left = {self.left}, right ="
            f" {self.right})"
        )

    def get_operands(self) -> list[VarBase]:
        return [self.left, self.right]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.left = replace_if_matched(self.left, old, new)
        self.right = replace_if_matched(self.right, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            left=replace_if_matched(self.left, old, new),
            right=replace_if_matched(self.right, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            left=self.left,
            right=self.right,
        )

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        # Now only works for vector <- matrix * vector specifically for RGAT linear operator reorder,
        # Or matrix <- matrix * matrix specifically for HGT linear operator reorder
        if curr_res_shape_info[0] is None:
            raise NotImplementedError(
                "Currently we only support WeightDenseOp that is generated"
                " during linear reorder pass. In other words, all the other"
                " shapes, especially the result shape of the operator should"
                " be already known."
            )
        if curr_res_shape_info[0].type == "vector":
            return [Shape.get_vector_shape(), Shape.get_matrix_shape()], [
                Shape.get_vector_shape()
            ]
        elif curr_res_shape_info[0].type == "matrix":
            return [Shape.get_matrix_shape(), Shape.get_matrix_shape()], [
                Shape.get_matrix_shape()
            ]
        else:
            raise NotImplementedError(
                "Currently we only support vector <- matrix * vector or matrix"
                " <- matrix * matrix specifically for RGAT linear operator"
                " reorder or HGT linear operator reorder."
            )


class EdgeDenseOp(_EdgeDenseOp, OpBase, metaclass=FinalOpMeta):
    # result: DataVar
    # input: DataVar
    # weight: WeightVar

    def validate(self) -> None:
        # Delta of weight is not possible to be the output: it needs outer-product
        assert isinstance(self.result, DataVar)
        assert isinstance(self.input, DataVar)
        assert isinstance(self.weight, WeightVar)
        self.result.validate()
        self.input.validate()
        self.weight.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]) -> "EdgeDenseOp":
        result = DataVar.from_string(d["result"])
        input = DataVar.from_string(d["input"])
        weight = WeightVar.from_string(d["weight"])
        return cls(result=result, input=input, weight=weight)

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "input": self.input.to_string(),
            "weight": self.weight.to_string(),
        }

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def lower(self) -> bool:
        raise NotImplementedError

    def to_string(self) -> str:
        return (
            f"{self.result}={self.get_opname()}(input = {self.input}, weight ="
            f" {self.weight})"
        )

    def get_operands(self) -> list[VarBase]:
        return [self.input, self.weight]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.input = replace_if_matched(self.input, old, new)
        self.weight = replace_if_matched(self.weight, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            input=replace_if_matched(self.input, old, new),
            weight=replace_if_matched(self.weight, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=self.input,
            weight=self.weight,
        )

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_matrix_shape()], [
            Shape.get_vector_shape()
        ]


class EdgeScalarVectorMulOp(
    _EdgeScalarVectorMulOp, OpBase, metaclass=FinalOpMeta
):
    # result: DataVar
    # scalar: DataVar
    # vector: DataVar

    def validate(self) -> None:
        assert isinstance(self.result, DataVar)
        assert isinstance(self.scalar, DataVar)
        assert isinstance(self.vector, DataVar)
        self.result.validate()
        self.scalar.validate()
        self.vector.validate()

    @classmethod
    def from_keyval_pairs(
        cls, d: dict["str", "str"]
    ) -> "EdgeScalarVectorMulOp":
        result = DataVar.from_string(d["result"])
        scalar = DataVar.from_string(d["scalar"])
        vector = DataVar.from_string(d["vector"])
        return cls(result=result, scalar=scalar, vector=vector)

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "scalar": self.scalar.to_string(),
            "vector": self.vector.to_string(),
        }

    def to_string(self) -> str:
        return (
            f"{self.result}={self.get_opname()}(scalar = {self.scalar}, vector"
            f" = {self.vector})"
        )

    def get_operands(self) -> list[VarBase]:
        return [self.scalar, self.vector]

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_scalar_shape(), Shape.get_vector_shape()], [
            Shape.get_vector_shape()
        ]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.scalar = replace_if_matched(self.scalar, old, new)
        self.vector = replace_if_matched(self.vector, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            scalar=replace_if_matched(self.scalar, old, new),
            vector=replace_if_matched(self.vector, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            scalar=self.scalar,
            vector=self.vector,
        )


class UnaryOp(_UnaryOp, OpBase, metaclass=FinalOpMeta):
    # result: VarBase
    # input: VarBase

    def validate(self) -> None:
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.input, (DataVar, WeightVar))
        self.result.validate()
        self.input.validate()

    @classmethod
    def from_keyval_pairs(cls: Type[T], d: dict["str", "str"]) -> T:
        result_cls = parse_var_class(d["result"])
        input_cls = parse_var_class(d["input"])
        result = result_cls.from_string(d["result"])
        input = input_cls.from_string(d["input"])
        return cls(result=result, input=input)

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "input": self.input.to_string(),
        }

    def to_string(self) -> str:
        return f"{self.result}={self.get_opname()}(input = {self.input})"

    def get_operands(self) -> list[VarBase]:
        return [self.input]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.input = replace_if_matched(self.input, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            input=replace_if_matched(self.input, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=self.input,
        )

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        # The input and the output should be the same shape
        curr_shape: str | None = None
        for ele in curr_opr_shape_info + curr_res_shape_info:
            if ele is not None:
                curr_shape = ele.type
                break
        assert curr_shape is not None
        return [Shape(curr_shape)], [Shape(curr_shape)]


class BinaryOp(_BinaryOp, OpBase, metaclass=FinalOpMeta):
    # result: VarBase
    # left: VarBase
    # right: VarBase

    def validate(self) -> None:
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.left, (DataVar, WeightVar))
        assert isinstance(self.right, (DataVar, WeightVar))
        self.result.validate()
        self.left.validate()
        self.right.validate()

    @classmethod
    def from_keyval_pairs(cls: Type[T], d: dict["str", "str"]) -> T:
        result_cls = parse_var_class(d["result"])
        left_cls = parse_var_class(d["left"])
        right_cls = parse_var_class(d["right"])
        result = result_cls.from_string(d["result"])
        left = left_cls.from_string(d["left"])
        right = right_cls.from_string(d["right"])
        return cls(result=result, left=left, right=right)

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "left": self.left.to_string(),
            "right": self.right.to_string(),
        }

    def to_string(self) -> str:
        return (
            f"{self.result}={self.get_opname()}(left = {self.left}, right ="
            f" {self.right})"
        )

    def get_operands(self) -> list[VarBase]:
        return [self.left, self.right]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def inplace_replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.left = replace_if_matched(self.left, old, new)
        self.right = replace_if_matched(self.right, old, new)

    def inplace_replace_all_results_with(
        self: ..., old: VarBase, new: VarBase
    ) -> None:
        self.result = replace_if_matched(self.result, old, new)

    def replace_all_operands_with(
        self: ..., old: VarBase, new: VarBase
    ) -> ...:
        return self.__class__(
            result=self.result,
            left=replace_if_matched(self.left, old, new),
            right=replace_if_matched(self.right, old, new),
        )

    def replace_all_results_with(self: ..., old: VarBase, new: VarBase) -> ...:
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            left=self.left,
            right=self.right,
        )

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        raise NotImplementedError


#
# Unary ops
#
class NodeSumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class TanhOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class InverseTanhOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class CopyOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class NegativeOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class EdgeTypeSumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class ExponentialOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class InverseExponentialOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class LeakyReluOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class InverseLeakyReluOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class SumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError


class TransposeOp(UnaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_matrix_shape()], [Shape.get_matrix_shape()]


#
# Binary ops
#
class ConcatenateOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        # The three shape should be the same
        curr_shape: str | None = None
        for ele in curr_opr_shape_info + curr_res_shape_info:
            if ele is not None:
                curr_shape = ele.type
                break
        assert curr_shape is not None
        return [Shape(curr_shape)] * len(curr_opr_shape_info), [
            Shape(curr_shape)
        ]


class VectorAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_vector_shape()], [
            Shape.get_vector_shape()
        ]


class EdgeInnerProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_vector_shape()], [
            Shape.get_scalar_shape()
        ]


class ScalarDivideOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_scalar_shape(), Shape.get_scalar_shape()], [
            Shape.get_scalar_shape()
        ]


class ScalarMultiplyOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_scalar_shape(), Shape.get_scalar_shape()], [
            Shape.get_scalar_shape()
        ]


class ScalarAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_scalar_shape(), Shape.get_scalar_shape()], [
            Shape.get_scalar_shape()
        ]


class EdgeOuterProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        # TODO: this is fusable with EdgeTypeSumAccumulationOp
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_vector_shape()], [
            Shape.get_matrix_shape()
        ]


class MatrixAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_matrix_shape(), Shape.get_matrix_shape()], [
            Shape.get_matrix_shape()
        ]


class NodeOuterProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError

    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        return [Shape.get_vector_shape(), Shape.get_vector_shape()], [
            Shape.get_matrix_shape()
        ]


class UnrealizedBinaryOp(BinaryOp, metaclass=abc.ABCMeta):
    """An abstract class from which UnrealizedAddOp and UnrealizedMulOp are inherited, so that we may use isinstance/issubclass to check whether an op is unrealized."""

    @abc.abstractmethod
    def realize(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> BinaryOp:
        raise NotImplementedError


class UnrealizedMulOp(UnrealizedBinaryOp):
    """Op that should be concretized to EdgeInnerProduct, ScalarMultiply, EdgeScalarVectorMul.
    We temporarily lower InterOpDSL to unrealized operators. After shape inference, we can tell what operators they really are.
    """

    def differentiate(self) -> list["OpBase"]:
        raise ValueError("UnrealizedMul should not be differentiated")

    def fusable_with(self, other: "OpBase") -> bool:
        raise ValueError("UnrealizedMul should not be fused")

    def lower(self):
        raise ValueError("UnrealizedMul should not be lowered")

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        num_scalar = 0
        num_vector = 0
        for ele in curr_opr_shape_info + curr_res_shape_info:
            if ele is not None:
                if ele.type == "scalar":
                    num_scalar += 1
                elif ele.type == "vector":
                    num_vector += 1
                else:
                    raise ValueError(
                        (
                            "unexpected shape during"
                            " UnrealizedMulOp.infer_shape()"
                        ),
                        ele.type,
                    )
        if num_scalar >= 2:
            return [Shape.get_scalar_shape()] * 2, [Shape.get_scalar_shape()]
        elif num_vector == 2:
            if (
                curr_res_shape_info[0] is not None
                and curr_res_shape_info[0].type == "vector"
            ):
                # EdgeScalarVectorMul
                return [Shape.get_scalar_shape(), Shape.get_vector_shape()], [
                    Shape.get_vector_shape()
                ]
            else:
                # EdgeInnerProduct
                return [Shape.get_vector_shape(), Shape.get_vector_shape()], [
                    Shape.get_scalar_shape()
                ]
        logger.warning("Insufficient information to infer shape of mul op")
        return curr_opr_shape_info, curr_res_shape_info

    def realize(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> BinaryOp | EdgeScalarVectorMulOp:
        """Realize the unrealized mul op to a concrete one. In current cases, it should be one among EdgeInnerProductOp, ScalarMultiplyOp, EdgeScalarVectorMulOp."""
        num_scalar = 0
        num_vector = 0

        # Determine the operator type by checking the number of scalar/vector in lhs and rhs
        for shape in curr_opr_shape_info:
            assert shape is not None
            if shape.type == "scalar":
                num_scalar += 1
            elif shape.type == "vector":
                num_vector += 1
            raise ValueError(
                "unexpected shape during UnrealizedMulOp.realize()", shape.type
            )
        if num_vector == 2:
            return EdgeInnerProductOp.from_keyval_pairs(self.to_keyval_pairs())
        elif num_vector == 1 and num_scalar == 1:
            return EdgeScalarVectorMulOp.from_keyval_pairs(
                self.to_keyval_pairs()
            )
        elif num_scalar == 2:
            return ScalarMultiplyOp.from_keyval_pairs(self.to_keyval_pairs())
        raise ValueError(
            "unexpected shape during UnrealizedMulOp.realize()",
            num_scalar,
            num_vector,
        )


class UnrealizedAddOp(UnrealizedBinaryOp):
    """Op that should be concretized to ScalarAdd, MatrixAdd, VectorAdd.
    We temporarily lower InterOpDSL to unrealized operators. After shape inference, we can tell what operators they really are.
    """

    def differentiate(self) -> list["OpBase"]:
        raise ValueError("UnrealizedAdd should not be differentiated")

    def fusable_with(self, other: "OpBase") -> bool:
        raise ValueError("UnrealizedAdd should not be fused")

    def lower(self):
        raise ValueError("UnrealizedAdd should not be lowered")

    def infer_shape(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> tuple[list[Shape | None], list[Shape | None]]:
        # The three shape should be the same
        curr_shape: str | None = None
        for ele in curr_opr_shape_info + curr_res_shape_info:
            if ele is not None:
                curr_shape = ele.type
                break
        assert curr_shape is not None
        return [Shape(curr_shape), Shape(curr_shape)], [Shape(curr_shape)]

    def realize(
        self,
        curr_opr_shape_info: list[Shape | None],
        curr_res_shape_info: list[Shape | None],
    ) -> BinaryOp:
        """Realize the unrealized add op to a concrete one. In current cases, it should be one among ScalarAddOp, VectorAddOp, MatrixAddOp."""
        # Determine the operator type by checking the shape of lhs and rhs
        result_shape: None | str = None
        for shape in curr_opr_shape_info:
            assert shape is not None
            if result_shape is None:
                result_shape = shape.type
            else:
                assert result_shape == shape.type
        if result_shape == "scalar":
            return ScalarAddOp.from_keyval_pairs(self.to_keyval_pairs())
        elif result_shape == "vector":
            return VectorAddOp.from_keyval_pairs(self.to_keyval_pairs())
        elif result_shape == "matrix":
            return MatrixAddOp.from_keyval_pairs(self.to_keyval_pairs())
        raise ValueError(
            "unexpected shape during UnrealizedAddOp.realize()", result_shape
        )


func_name_to_op: dict[str, Type[OpBase]] = {
    "Split": SplitOp,  # (results) input
    "NodeDense": NodeDenseOp,  # input, weight
    "WeightDense": WeightDenseOp,  # left, right
    "EdgeDense": EdgeDenseOp,  # input, weight
    "EdgeScalarVectorMul": EdgeScalarVectorMulOp,  # scalar, vector
    # Unary Ops. keyword: input
    "NodeSumAccumulation": NodeSumAccumulationOp,
    "Tanh": TanhOp,
    "InverseTanh": InverseTanhOp,
    "Copy": CopyOp,
    "Negative": NegativeOp,
    "EdgeTypeSumAccumulation": EdgeTypeSumAccumulationOp,
    "Exponential": ExponentialOp,
    "InverseExponential": InverseExponentialOp,
    "LeakyRelu": LeakyReluOp,
    "InverseLeakyRelu": InverseLeakyReluOp,
    "SumAccumulation": SumAccumulationOp,
    "Transpose": TransposeOp,
    # Binary Ops. keyword: left, right
    "VectorAdd": VectorAddOp,
    "Concatenate": ConcatenateOp,
    "EdgeInnerProduct": EdgeInnerProductOp,
    "ScalarDivide": ScalarDivideOp,
    "ScalarMultiply": ScalarMultiplyOp,
    "ScalarAdd": ScalarAddOp,
    "EdgeOuterProduct": EdgeOuterProductOp,
    "MatrixAdd": MatrixAddOp,
    "NodeOuterProduct": NodeOuterProductOp,
}


# TODO: implement the following two fuse ops


class TraversalFusedOp(FusedOpBase):
    @classmethod
    def get_opname(cls) -> str:
        return "TraversalOp"

    def differentiate(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def lower(self):
        raise NotImplementedError


class GEMMFusedOp(FusedOpBase):
    @classmethod
    def get_opname(cls) -> str:
        return "GEMMOp"

    def differentiate(self):
        raise NotImplementedError

    def validate(self) -> None:
        raise NotImplementedError

    def lower(self):
        raise NotImplementedError


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    a = DataVar.from_string('(NODEWISE,"a")')
    print(a.to_string())
    b = SplitOp.from_keyval_pairs(
        {
            "func_name": "Split",
            "results": [
                '(NODEWISE,"a")',
                '(EDGEWISE,"b")',
            ],
            "input": '(NODEWISE,"c")',
        }
    )
    c = UnrealizedAddOp()
    print(issubclass(UnrealizedAddOp, BinaryOp))
    print(issubclass(UnrealizedAddOp, UnrealizedBinaryOp))
    print(b.results[0].to_string())
    print(b.to_keyval_pairs())
    print(b.to_keyval_pairs.__name__)
    print(isinstance(b, _SplitOp))
    print(TraversalFusedOp.from_ops(b.results, [b.input], [b, b]).to_string())
