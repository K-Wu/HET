import abc
from recordclass import dataobject
from ..InterOpSSA.variables import WeightVar, DataVar, parse_var_spec_class
import json
from typing import Any, Union, Optional


class OpSpecBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def to_opspec_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_opspec_dict(cls, opspec_dict: dict[str, Any]) -> "OpSpecBase":
        raise NotImplementedError


class FinalOpSpecMeta(type(dataobject), type(OpSpecBase)):
    pass


class TraversalSimpleOpSpec(dataobject):
    op: str  # e.g., "expf"
    # op_type example "elementwise", {"type":"accumulation", "direction":"node"}
    op_type: Optional[str | dict[str, str]]
    broadcast: Optional[dict[str, str]]
    inputs: list[Union[DataVar, WeightVar]]
    output: Union[DataVar, WeightVar]

    def to_opspec_dict(self) -> dict[str, Any]:
        result = {
            "op": self.op,
            "inputs": [ele.to_opspec_list() for ele in self.inputs],
            "output": self.output.to_opspec_list(),
        }
        if self.op_type is not None:
            result["op_type"] = self.op_type
        if self.broadcast is not None:
            result["broadcast"] = self.broadcast
        return result

    @classmethod
    def from_opspec_dict(
        cls, opspec_dict: dict[str, Any]
    ) -> "TraversalSimpleOpSpec":
        inputs = opspec_dict["inputs"]
        return cls(
            op=opspec_dict["op"],
            op_type=opspec_dict.get("op_type"),
            broadcast=opspec_dict.get("broadcast"),
            inputs=[
                parse_var_spec_class(ele).from_opspec_list(ele)
                for ele in inputs
            ],
            output=parse_var_spec_class(
                opspec_dict["output"]
            ).from_opspec_list(opspec_dict["output"]),
        )


class TraversalLoopOpSpec(dataobject):
    loop_variable: str
    loop_begin: str
    loop_end: str
    loop_step: str
    # Each element in loop_scalar_tmps is a list of two strings describing the type and name of the temporary scalar variable
    loop_scalar_tmps: list[list[str]]
    operators: list["TraversalSimpleOpSpec | TraversalLoopOpSpec"]

    def to_opspec_dict(self) -> dict[str, Any]:
        result = {
            "loop_variable": self.loop_variable,
            "loop_begin": self.loop_begin,
            "loop_end": self.loop_end,
            "loop_step": self.loop_step,
            "loop_scalar_tmps": self.loop_scalar_tmps,
            "operators": {
                "{idx}:{type}".format(
                    idx=idx + 1,
                    type=(
                        "loop"
                        if isinstance(operator, TraversalLoopOpSpec)
                        else "op"
                    ),
                ): operator.to_opspec_dict()
                for idx, operator in enumerate(self.operators)
            },
        }
        return result

    @classmethod
    def from_opspec_dict(
        cls, opspec_dict: dict[str, Any]
    ) -> "TraversalLoopOpSpec":
        return cls(
            loop_variable=opspec_dict["loop_variable"],
            loop_begin=opspec_dict["loop_begin"],
            loop_end=opspec_dict["loop_end"],
            loop_step=opspec_dict["loop_step"],
            loop_scalar_tmps=opspec_dict["loop_scalar_tmps"],
            operators=[
                item[1]
                for item in sorted(
                    opspec_dict["operators"].items(),
                    lambda x: int(x[0].split(":")[0]),
                )
            ],
        )


class _TraversalOpSpec(dataobject):
    outputs: list[Union[DataVar, WeightVar]]
    inputs: list[Union[DataVar, WeightVar]]
    op_idx: int
    operators: list[TraversalSimpleOpSpec | TraversalLoopOpSpec]
    schedule: str  # "type1" or "type2"
    # Type 1 Schedule:
    # head -> blockIdx.x * blockDim.x + threadIdx.x;
    # edge|node -> blockIdx.y * blockDim.y + threadIdx.y;
    # Type 2 Schedule:
    # head -> threadIdx.y
    # edge|node -> blockIdx.y
    # feat_idx -> blockIdx.x * blockDim.x + threadIdx.x


class TraversalOpSpec(_TraversalOpSpec, OpSpecBase, metaclass=FinalOpSpecMeta):
    def to_opspec_dict(self) -> dict[str, Any]:
        opspec_dict: dict[str, Any] = {
            "outputs": [ele.to_opspec_list() for ele in self.outputs],
            "inputs": [ele.to_opspec_list() for ele in self.inputs],
            "schedule": self.schedule,
            "op_idx": self.op_idx,
            "operators": {
                "{idx}:{type}".format(
                    idx=idx + 1, type=operator.type
                ): operator.to_opspec_dict()
                for idx, operator in enumerate(self.operators)
            },
        }
        return opspec_dict

    @classmethod
    def from_opspec_dict(
        cls, opspec_dict: dict[str, Any]
    ) -> "TraversalOpSpec":
        inputs = [
            parse_var_spec_class(ele).from_opspec_list(ele)
            for ele in opspec_dict["inputs"]
        ]
        outputs = [
            parse_var_spec_class(ele).from_opspec_list(ele)
            for ele in opspec_dict["outputs"]
        ]
        return cls(
            outputs=outputs,
            inputs=inputs,
            op_idx=opspec_dict["op_idx"],
            operators=[
                item[1]
                for item in sorted(
                    opspec_dict["operators"].items(),
                    lambda x: int(x[0].split(":")[0]),
                )
            ],
            schedule=opspec_dict["schedule"],
        )

    def to_string(self) -> str:
        return (
            f"traversal_{self.op_idx}"
            + "{\n"
            + json.dumps(self.to_opspec_dict())
            + "\n}"
        )


class _GEMMOpSpec(dataobject):
    product: Union[DataVar, WeightVar]
    left: Union[DataVar, WeightVar]
    right: Union[DataVar, WeightVar]
    op_idx: int
    edgewise_use_compaction: dict[Union[DataVar, WeightVar], bool]
    schedule: dict[str, Any]
    access: dict[str, Any]


class GEMMOpSpec(_GEMMOpSpec, OpSpecBase, metaclass=FinalOpSpecMeta):
    def to_opspec_dict(self) -> dict[str, Any]:
        opspec_dict: dict[str, Any] = {
            "product": list(self.product.to_opspec_list()),
            "left": list(self.left.to_opspec_list()),
            "right": list(self.right.to_opspec_list()),
            "op_idx": self.op_idx,
            "edgewise_use_compaction": self.edgewise_use_compaction,
            "schedule": self.schedule,
            "access": self.access,
        }
        return opspec_dict

    @classmethod
    def from_opspec_dict(cls, opspec_dict: dict[str, Any]) -> "GEMMOpSpec":
        left = parse_var_spec_class(opspec_dict["left"]).from_opspec_list(
            opspec_dict["left"]
        )
        right = parse_var_spec_class(opspec_dict["right"]).from_opspec_list(
            opspec_dict["right"]
        )
        product = parse_var_spec_class(
            opspec_dict["product"]
        ).from_opspec_list(opspec_dict["product"])
        return cls(
            product=product,
            left=left,
            right=right,
            op_idx=opspec_dict["op_idx"],
            edgewise_use_compaction=opspec_dict["edgewise_use_compaction"],
            schedule=opspec_dict["schedule"],
            access=opspec_dict["access"],
        )

    def to_string(self) -> str:
        return (
            f"gemm_{self.op_idx}"
            + "{\n"
            + json.dumps(self.to_opspec_dict())
            + "\n}"
        )
