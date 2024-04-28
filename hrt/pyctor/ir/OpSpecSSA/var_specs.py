from ..InterOpSSA.variables import (
    DataVar,
    WeightVar,
    is_valid_var_name,
    DATA_TYPES,
)
from typing import NamedTuple, Union, Type

_DataVarSpec = NamedTuple(
    "DataVarSpec", [("type", str), ("name", str), ("compaction", bool)]
)
DATA_SPEC_TYPES = DATA_TYPES.copy()
assert len(DATA_SPEC_TYPES) == 4
DATA_SPEC_TYPES["UNIQUE_NODE_ETYPE"] = 4


class DataVarSpec(_DataVarSpec):
    """Add compaction flag on top of DataVar and support importing from OPSPEC and exporting to OPSPEC"""

    @classmethod
    def serialize_type_and_compaction(cls, type: str, compaction: bool) -> str:
        if compaction:
            return f"UNIQUE_NODE_ETYPE"
        else:
            return type

    @classmethod
    def deserialize_type_and_compaction(cls, type: str) -> tuple[str, bool]:
        if type == "UNIQUE_NODE_ETYPE":
            return "EDGEWISE", True
        else:
            return type, False

    @classmethod
    def from_string(cls, s: str) -> "DataVarSpec":
        keyval_str = s.split(",")
        assert len(keyval_str) == 2
        keyval_str = [ele.strip() for ele in keyval_str]

        # removing parantheses and quotes
        type = keyval_str[0][1:]
        name = keyval_str[1][1:-2]

        # Handle compaction
        type, compaction = cls.deserialize_type_and_compaction(type)
        return cls(type=type, name=name, compaction=compaction)

    @classmethod
    def from_dict(cls, d: dict["str", "str"]) -> "DataVarSpec":
        type, compaction = cls.deserialize_type_and_compaction(d["type"])
        return cls(type=type, name=d["name"], compaction=compaction)

    @classmethod
    def from_list(cls, l: list[str]) -> "DataVarSpec":
        """This method is provided to support importing from OPSPEC"""
        type, compaction = cls.deserialize_type_and_compaction(l[0])
        return cls(type=type, name=l[1], compaction=compaction)

    def to_list(self) -> list[str]:
        """This method is provided to support exporting to OPSPEC"""
        # type = self.serialize_type_and_compaction(self.type, self.compaction)
        # return [type, self.name]
        return list(self.to_dict().values())

    def to_dict(self) -> dict["str", "str"]:
        type = self.serialize_type_and_compaction(self.type, self.compaction)
        return {"type": type, "name": self.name}

    def to_string(self) -> str:
        type = self.serialize_type_and_compaction(self.type, self.compaction)
        return f'({type}, "{self.name}")'

    def get_name(self) -> str:
        return self.name

    def validate(self) -> None:
        assert self.type in DATA_SPEC_TYPES
        assert is_valid_var_name(self.name)


def parse_var_spec_class(
    name: str,
) -> Union[Type[DataVarSpec], Type[WeightVar]]:
    if '"' in name:
        return DataVarSpec
    else:
        return WeightVar
