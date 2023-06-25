#!/usr/bin/env python3
from .variables import *
from typing import NamedTuple, Union
import abc


# when inherited, OpBase provides the interface of operator classes
class OpBase(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]):
        # from_keyval_pairs takes in keyval pairs parsed by op_serializer.py
        # recursively rather instantiated object as keys' values by cls._make(d)
        raise NotImplementedError

    @abc.abstractmethod
    def to_keyval_pairs(self) -> dict["str", "str"]:
        # to_keyval_pairs returns dict of strings recursively rather than
        # instantiated objects as keys' values self._asdict()
        raise NotImplementedError

    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fusable_with(self, other: "OpBase") -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def differentiate(self) -> bool:
        raise NotImplementedError


_SplitOp = NamedTuple("SplitOp", [("results", list[DataVar]), ("input", DataVar)])
_ConcatenateOp = NamedTuple(
    "ConcatenateOp", [("result", DataVar), ("input", DataVar)]
)  # input is a list
_NodeLinearOp = NamedTuple(
    "NodeLinearOp", [("result", DataVar), ("input", DataVar), ("weight", WeightVar)]
)
_EdgeLinearOp = NamedTuple(
    "EdgeLinearOp", [("result", DataVar), ("input", DataVar), ("weight", WeightVar)]
)
_EdgeScalarVectorMulOp = NamedTuple(
    "EdgeScalarVectorMulOp",
    [("result", DataVar), ("scalar", DataVar), ("vector", DataVar)],
)
_UnaryOp = NamedTuple(
    "_UnaryOp",
    [("result", Union[DataVar, WeightVar]), ("input", Union[DataVar, WeightVar])],
)
_BinaryOp = NamedTuple(
    "_BinaryOp",
    [
        ("result", Union[DataVar, WeightVar]),
        ("left", Union[DataVar, WeightVar]),
        ("right", Union[DataVar, WeightVar]),
    ],
)


#
# Operators that neither fits in ordinary unary ops' fields nor binary ops'
# fields
#
class ConcatenateOp(_ConcatenateOp, OpBase):
    def validate(self):
        raise NotImplementedError


class SplitOp(_SplitOp, OpBase):
    @classmethod
    def from_keyval_pairs(cls, d: dict["str", Union["str", list["str"]]]):
        results = []
        for ele in d["results"]:
            results.append(DataVar.from_string(ele))
        assert isinstance(d["input"], str)
        input = DataVar.from_string(d["input"])
        return cls(results=results, input=input)

    def to_keyval_pairs(self) -> dict["str", Union["str", list["str"]]]:
        return {
            "results": [ele.to_string() for ele in self.results],
            "input": self.input.to_string(),
        }

    def validate(self):
        raise NotImplementedError


class NodeLinearOp(_NodeLinearOp, OpBase):
    def validate(self):
        raise NotImplementedError


class EdgeLinearOp(_EdgeLinearOp, OpBase):
    def validate(self):
        raise NotImplementedError


class EdgeScalarVectorMulOp(_EdgeScalarVectorMulOp, OpBase):
    def validate(self):
        raise NotImplementedError


class UnaryOp(_UnaryOp, OpBase):
    def validate(self):
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.input, (DataVar, WeightVar))
        self.result.validate()
        self.input.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]):
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


class BinaryOp(_BinaryOp, OpBase):
    def validate(self):
        assert isinstance(self.result, (DataVar, WeightVar))
        assert isinstance(self.left, (DataVar, WeightVar))
        assert isinstance(self.right, (DataVar, WeightVar))
        self.result.validate()
        self.left.validate()
        self.right.validate()

    @classmethod
    def from_keyval_pairs(cls, d: dict["str", "str"]):
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


#
# Unary ops
#
class NodeSumAccumulationOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class TanhOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class InverseTanhOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class CopyOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class NegativeOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class EdgeTypeSumAccumulationOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class ExponentialOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class InverseExponentialOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class LeakyReluOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class InverseLeakyReluOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class SumAccumulationOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


class TransposeOp(UnaryOp):
    def validate(self):
        raise NotImplementedError


#
# Binary ops
#
class VectorAddOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class EdgeInnerProductOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class ScalarDevideOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class ScalarMultiplyOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class ScalarAddOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class EdgeOuterProductOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class MatrixAddOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


class NodeOuterProductOp(BinaryOp):
    def validate(self):
        raise NotImplementedError


name_to_op_mapper = {
    "Concatenate": ConcatenateOp,  # input (is a list)
    "Split": SplitOp,  # (results) input
    "NodeLinear": NodeLinearOp,  # input, weight
    "EdgeLinear": EdgeLinearOp,  # input, weight
    "EdgeScalarVectorMul": EdgeLinearOp,  # scalar, vector
    # Unary Ops. keyword: input
    "NodeSumAccumulation": NodeLinearOp,
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
    "EdgeInnerProduct": EdgeInnerProductOp,
    "ScalarDevide": ScalarDevideOp,
    "ScalarMultiply": ScalarMultiplyOp,
    "ScalarAdd": ScalarAddOp,
    "EdgeOuterProduct": EdgeOuterProductOp,
    "MatrixAdd": MatrixAddOp,
    "NodeOuterProduct": NodeOuterProductOp,
}


# returns True if 1) every operation has all key-value pairs correctly
# defined as specified in this file, and 2) use-def chain is correct
def validate():
    raise NotImplementedError


if __name__ == "__main__":
    a = DataVar.from_string('(NODEWISE,"a")')
    print(a.to_string())
    b = SplitOp.from_keyval_pairs(
        {
            "results": [
                '(NODEWISE,"a")',
                '(EDGEWISE,"b")',
            ],
            "input": '(NODEWISE,"c")',
        }
    )
    print(b.results[0].to_string())
    print(b.to_keyval_pairs())
