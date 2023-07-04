#!/usr/bin/env python3
from .variables import VarBase, DataVar, WeightVar, get_var_class
from typing import NamedTuple, Union, Type, TypeVar
import abc

T = TypeVar("T")


class OpBase(metaclass=abc.ABCMeta):
    """when inherited, OpBase provides the interface of operator classes"""

    @classmethod
    def get_opname(cls) -> str:
        return cls.__name__[:-2]  # remove "Op" suffix

    @classmethod
    @abc.abstractmethod
    def from_keyval_pairs(cls: Type[T], d: dict["str", Union[list["str"], "str"]]) -> T:
        """
        from_keyval_pairs takes in keyval pairs parsed by op_serializer.py
        recursively rather instantiated object as keys' values by cls._make(d)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_keyval_pairs(self) -> dict["str", "str"]:
        """
        to_keyval_pairs returns dict of strings recursively rather than
        instantiated objects as keys' values self._asdict()
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

    @abc.abstractclassmethod
    def get_operands(self) -> list[VarBase]:
        raise NotImplementedError

    @abc.abstractclassmethod
    def get_results(self) -> list[VarBase]:
        raise NotImplementedError

    @abc.abstractclassmethod
    def replace_var_with(self: T, old: VarBase, new: VarBase) -> T:
        raise NotImplementedError


class FusedOpBase(metaclass=abc.ABCMeta):
    ops: list[OpBase]

    def __init__(self, ops: list[OpBase]):
        self.ops = ops

    @classmethod
    def get_opname(cls) -> str:
        raise NotImplementedError

    @classmethod
    def from_ops(cls, ops: list[OpBase]):
        return cls(ops)

    def to_string(self) -> str:
        result = self.get_opname() + "{\n"
        for op in self.ops:
            result += op.to_string()
            result += "\n"
        result += "}"
        return result

    @abc.abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def differentiate(self) -> list["OpBase"]:
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self) -> bool:
        raise NotImplementedError

    def replace_var_with(self: T, old: VarBase, new: VarBase) -> T:
        raise NotImplementedError


_SplitOp = NamedTuple("SplitOp", [("results", list[DataVar]), ("input", DataVar)])
_NodeDenseOp = NamedTuple(
    "NodeDenseOp", [("result", DataVar), ("input", DataVar), ("weight", WeightVar)]
)
_EdgeDenseOp = NamedTuple(
    "EdgeDenseOp", [("result", DataVar), ("input", DataVar), ("weight", WeightVar)]
)
_EdgeScalarVectorMulOp = NamedTuple(
    "EdgeScalarVectorMulOp",
    [("result", DataVar), ("scalar", DataVar), ("vector", DataVar)],
)
_UnaryOp = NamedTuple(
    "_UnaryOp",
    [("result", VarBase), ("input", VarBase)],
)
_BinaryOp = NamedTuple(
    "_BinaryOp",
    [
        ("result", VarBase),
        ("left", VarBase),
        ("right", VarBase),
    ],
)


def replace_if_matched(var: VarBase, old: VarBase, new: VarBase) -> ...:
    if var == old:
        return new
    else:
        return var


#
# Operators that neither fits in ordinary unary ops' fields nor binary ops'
# fields
#
class SplitOp(_SplitOp, OpBase):
    def validate(self) -> None:
        for ele in self.results:
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

    def to_string(self) -> str:
        results_str = ",".join([ele.to_string() for ele in self.results])
        return f"{results_str}={self.get_opname()}(input = {self.input})"

    def get_operands(self) -> list[VarBase]:
        return [self.input]

    def get_results(self) -> list[VarBase]:
        # Unpack and pack again to avoid type error
        return [*self.results]

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            results=[replace_if_matched(ele, old, new) for ele in self.results],
            input=replace_if_matched(self.input, old, new),
        )


class NodeDenseOp(_NodeDenseOp, OpBase):
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

    def to_keyval_pairs(self) -> dict["str", "str"]:
        return {
            "result": self.result.to_string(),
            "input": self.input.to_string(),
            "weight": self.weight.to_string(),
        }

    def to_string(self) -> str:
        return f"{self.result}={self.get_opname()}(input = {self.input}, weight = {self.weight})"

    def get_operands(self) -> list[VarBase]:
        return [self.input, self.weight]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=replace_if_matched(self.input, old, new),
            weight=replace_if_matched(self.weight, old, new),
        )


class EdgeDenseOp(_EdgeDenseOp, OpBase):
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

    def to_string(self) -> str:
        return f"{self.result}={self.get_opname()}(input = {self.input}, weight = {self.weight})"

    def get_operands(self) -> list[VarBase]:
        return [self.input, self.weight]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=replace_if_matched(self.input, old, new),
            weight=replace_if_matched(self.weight, old, new),
        )


class EdgeScalarVectorMulOp(_EdgeScalarVectorMulOp, OpBase):
    def validate(self) -> None:
        assert isinstance(self.result, DataVar)
        assert isinstance(self.scalar, DataVar)
        assert isinstance(self.vector, DataVar)
        self.result.validate()
        self.scalar.validate()
        self.vector.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]) -> "EdgeScalarVectorMulOp":
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
        return f"{self.result}={self.get_opname()}(scalar = {self.scalar}, vector = {self.vector})"

    def get_operands(self) -> list[VarBase]:
        return [self.scalar, self.vector]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            scalar=replace_if_matched(self.scalar, old, new),
            vector=replace_if_matched(self.vector, old, new),
        )


class UnaryOp(_UnaryOp, OpBase):
    def validate(self) -> None:
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.input, (DataVar, WeightVar))
        self.result.validate()
        self.input.validate()

    @classmethod
    def from_keyval_pairs(cls: Type[T], d: dict["str", "str"]) -> T:
        result_cls = get_var_class(d["result"])
        input_cls = get_var_class(d["input"])
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

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            input=replace_if_matched(self.input, old, new),
        )


class BinaryOp(_BinaryOp, OpBase):
    def validate(self) -> None:
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.left, (DataVar, WeightVar))
        assert isinstance(self.right, (DataVar, WeightVar))
        self.result.validate()
        self.left.validate()
        self.right.validate()

    @classmethod
    def from_keyval_pairs(cls: Type[T], d: dict["str", "str"]) -> T:
        result_cls = get_var_class(d["result"])
        left_cls = get_var_class(d["left"])
        right_cls = get_var_class(d["right"])
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
        return f"{self.result}={self.get_opname()}(left = {self.left}, right = {self.right})"

    def get_operands(self) -> list[VarBase]:
        return [self.left, self.right]

    def get_results(self) -> list[VarBase]:
        return [self.result]

    def replace_var_with(self, old: VarBase, new: VarBase):
        return self.__class__(
            result=replace_if_matched(self.result, old, new),
            left=replace_if_matched(self.left, old, new),
            right=replace_if_matched(self.right, old, new),
        )


#
# Unary ops
#
class NodeSumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class TanhOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class InverseTanhOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class CopyOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class NegativeOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class EdgeTypeSumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class ExponentialOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class InverseExponentialOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class LeakyReluOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class InverseLeakyReluOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class SumAccumulationOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


class TransposeOp(UnaryOp):
    def lower(self):
        raise NotImplementedError


#
# Binary ops
#
class ConcatenateOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class VectorAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class EdgeInnerProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class ScalarDivideOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class ScalarMultiplyOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class ScalarAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class EdgeOuterProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class MatrixAddOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class NodeOuterProductOp(BinaryOp):
    def lower(self):
        raise NotImplementedError


class UnrealizedMulOp(BinaryOp):
    """Op that should be concretized to EdgeInnerProduct, ScalarMultiply, EdgeScalarVectorMul.
    We temporarily lower InterOpDSL to unrealized operators. After shape inference, we can tell what operators they really are.
    """

    def lower(self):
        raise ValueError("UnrealizedMul should not be lowered")


class UnrealizedAddOp(BinaryOp):
    """Op that should be concretized to ScalarAdd, MatrixAdd, VectorAdd.
    We temporarily lower InterOpDSL to unrealized operators. After shape inference, we can tell what operators they really are.
    """

    def lower(self):
        raise ValueError("UnrealizedAdd should not be lowered")


func_name_to_op: dict[str, Type[OpBase]] = {
    "Split": SplitOp,  # (results) input
    "NodeDense": NodeDenseOp,  # input, weight
    "EdgeDense": EdgeDenseOp,  # input, weight
    "EdgeScalarVectorMul": EdgeDenseOp,  # scalar, vector
    # Unary Ops. keyword: input
    "NodeSumAccumulation": NodeDenseOp,
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

    def validate(self):
        raise NotImplementedError

    def lower(self):
        raise NotImplementedError


if __name__ == "__main__":
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
    print(b.results[0].to_string())
    print(b.to_keyval_pairs())
    print(TraversalFusedOp.from_ops([b, b]).to_string())
