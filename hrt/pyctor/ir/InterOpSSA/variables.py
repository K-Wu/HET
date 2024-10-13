#!/usr/bin/env python3
from typing import Type, Union, NamedTuple
import abc

# TODO: Store dimension assignment to enhance extensibility
_Shape = NamedTuple("Shape", [("row_purpose", str), ("slice_type", str)])
SHAPE_ROW_PURPOSE_TYPES = {
    "nodewise": 0,
    "dstnode": 1,
    "srcnode": 2,
    "edgewise": 3,
    "unique_node_etype": 4,
    "edgetype": 5,
    "nodetype": 6,
    "none_as_weight_slice_type": 7,
    "unassigned": 99,
}
SHAPE_SLICE_TYPES = {"scalar": 0, "vector": 1, "matrix": 2}


class Shape(_Shape):
    @classmethod
    def from_string(cls, s: str) -> "Shape":
        keyval_str = s.split(",")
        assert len(keyval_str) == 2
        keyval_str = [ele.strip() for ele in keyval_str]
        # Remove bracket
        keyval_str[0] = keyval_str[0][1:]
        keyval_str[1] = keyval_str[1][:-1]
        return cls(row_purpose=keyval_str[0], slice_type=keyval_str[1])

    @classmethod
    def get_scalar_shape(cls) -> "Shape":
        # (idx_entry, idx_head)
        return cls(row_purpose="unassigned", slice_type="scalar")

    @classmethod
    def get_vector_shape(cls) -> "Shape":
        # (idx_entry, idx_head, idx_element)
        return cls(row_purpose="unassigned", slice_type="vector")

    @classmethod
    def get_matrix_shape(cls) -> "Shape":
        # (idx_entry, idx_head, idx_row, idx_column)
        return cls(row_purpose="unassigned", slice_type="matrix")

    @classmethod
    def from_dict(cls, d: dict["str", "str"]) -> "Shape":
        return cls(row_purpose="unassigned", slice_type=d["slice_type"])

    def validate(self) -> None:
        assert (
            self.row_purpose in SHAPE_ROW_PURPOSE_TYPES
            and self.slice_type in SHAPE_SLICE_TYPES
        )

    def to_dict(self) -> dict["str", "str"]:
        return {"row_purpose": self.row_purpose, "slice_type": self.slice_type}

    def to_string(self) -> str:
        return f"[{self.row_purpose}, {self.slice_type}]"


class VarBase(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_string(cls, s: str) -> "VarBase":
        # from_string instantiates an object based on input string as an
        # elementary procedure during from_keyval_pairs calls
        raise NotImplementedError

    @abc.abstractmethod
    def to_string(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_delta(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d: dict["str", "str"]) -> "VarBase":
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> dict["str", "str"]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_numbered_var(self, name: str) -> "VarBase":
        """preserve all the rest of the information, but change the name to name"""
        raise NotImplementedError


# Shape is stored separately from VarBase because it needs further inference and may be extended to support different dimension assignment in future
# TODO: In future, we may store separately the dtype like we do for shape to support type different from float32
_WeightVar = NamedTuple("WeightVar", [("name", str), ("slice_type", str)])
_DataVar = NamedTuple("DataVar", [("type", str), ("name", str)])

WEIGHT_SLICE_TYPES = {"EDGETYPE": 0, "NODETYPE": 1, "NONE": 2}
# TODO: Decouple UNIQUE_NODE_ETYPE from DATA_TYPES so that in DAG both UNIQUE_NODE_ETYPE and EDGEWIDE are shown as EDGEWISE while UNIQUE_NODE_ETYPE is recorded in the shape table and opspec ssa
DATA_TYPES = {
    "EDGEWISE": 0,
    "NODEWISE": 1,
    "DSTNODE": 2,
    "SRCNODE": 3,
    "UNIQUE_NODE_ETYPE": 4,
}


def is_valid_var_name(name: str) -> bool:
    # varname(_src|_dst)?(_tmp[0-9])?(_delta)?(_term[0-9])?(_tmp[0-9])?(_transposed)?
    # _term is reserved for auto-differentiation where multiple terms contribute to one gradient
    suffix_locations = []
    if "_src" in name:
        if name.count("_src") > 1:
            return False
        suffix_locations.append(name.find("_src"))
    if "_dst" in name:
        if name.count("_dst") > 1:
            return False
        if "_src" in name:
            return False
        suffix_locations.append(name.find("_dst"))
    if "_delta" in name:
        if name.count("_delta") > 1:
            return False
        # there could be delta of tmp variables, so we now only check if _delta is after _src or _dst
        if "_src" in name:
            if name.find("_delta") < name.find("_src"):
                return False
        if "_dst" in name:
            if name.find("_delta") < name.find("_dst"):
                return False
    if "_term" in name:
        if name.count("_term") > 1:
            return False
        suffix_locations.append(name.find("_term"))
    if "_tmp" in name:
        suffix_locations.append(name.find("_tmp"))
    if "_transposed" in name:
        if name.count("_transposed") > 1:
            return False
        suffix_locations.append(name.find("_transposed"))
    # check if suffix_locations are in ascending order
    return sorted(suffix_locations) == suffix_locations


# TODO: lower()
class WeightVar(_WeightVar, VarBase):
    # @classmethod
    # def from_keyval_pairs(cls, d: dict) -> "WeightVar":
    #     return cls(name=d["name"], slice_type=d["slice_type"])

    @classmethod
    def from_string(cls, s: str) -> "WeightVar":
        keyval_str = s.split(",")
        assert len(keyval_str) == 2
        keyval_str = [ele.strip() for ele in keyval_str]
        # removing parantheses
        return cls(name=keyval_str[0][1:], slice_type=keyval_str[1][:-1])

    # def to_keyval_pairs(self) -> dict:
    #    return {"name": self.name, "slice_type": self.slice_type}

    @classmethod
    def from_dict(cls, d: dict["str", "str"]) -> "WeightVar":
        return cls(name=d["name"], slice_type=d["slice_type"])

    @classmethod
    def from_opspec_list(cls, l: list[str]) -> "WeightVar":
        """This method is provided for opspec deserialization."""
        return cls(name=l[0], slice_type=l[1])

    def to_opspec_list(self) -> list[str]:
        """This method is provided for opspec serialization."""
        return [self.name, self.slice_type]

    def to_dict(self) -> dict["str", "str"]:
        return {"name": self.name, "slice_type": self.slice_type}

    def get_name(self) -> str:
        return self.name

    def to_string(self) -> str:
        return f"({self.name}, {self.slice_type})"

    def validate(self) -> None:
        assert self.slice_type in WEIGHT_SLICE_TYPES
        assert is_valid_var_name(self.name)

    def lower(self):
        raise NotImplementedError

    def is_delta(self) -> bool:
        return "_delta" in self.name

    def transpose(self) -> "WeightVar":
        if "_transposed" in self.name:
            new_name = self.name.replace("_transposed", "")
        else:
            new_name = self.name + "_transposed"
        return WeightVar(name=new_name, slice_type=self.slice_type)

    def differentiate(self) -> "WeightVar":
        return WeightVar(name=self.name + "_delta", slice_type=self.slice_type)

    def get_numbered_var(self, name: str) -> "WeightVar":
        return WeightVar(name=name, slice_type=self.slice_type)


class DataVar(_DataVar, VarBase):
    # @classmethod
    # def from_keyval_pairs(cls, d: dict) -> "DataVar":
    #     return cls(type=d["type"], name=d["name"])

    # def to_keyval_pairs(self) -> dict:
    #     return {"type": self.type, "name": self.name}

    @classmethod
    def from_string(cls, s: str) -> "DataVar":
        keyval_str = s.split(",")
        assert len(keyval_str) == 2
        keyval_str = [ele.strip() for ele in keyval_str]
        # removing parantheses and quotes
        return cls(type=keyval_str[0][1:], name=keyval_str[1][1:-2])

    @classmethod
    def from_opspec_list(cls, l: list[str]) -> "DataVar":
        """This method is provided for opspec deserialization."""
        return cls(type=l[0], name=l[1])

    def to_opspec_list(self) -> list[str]:
        """This method is provided for opspec serialization."""
        return [self.type, self.name]

    @classmethod
    def from_dict(cls, d: dict["str", "str"]) -> "DataVar":
        return cls(type=d["type"], name=d["name"])

    def to_dict(self) -> dict["str", "str"]:
        return {"type": self.type, "name": self.name}

    def to_string(self) -> str:
        return f'({self.type}, "{self.name}")'

    def get_name(self) -> str:
        return self.name

    def validate(self) -> None:
        assert self.type in DATA_TYPES
        assert is_valid_var_name(self.name)

    def lower(self):
        raise NotImplementedError

    def is_delta(self) -> bool:
        return "_delta" in self.name

    def differentiate(self):
        return DataVar(type=self.type, name=self.name + "_delta")

    def get_numbered_var(self, name: str) -> "DataVar":
        return DataVar(name=name, type=self.type)


def parse_var_spec_class(
    varspec: list[str],
) -> Union[Type[DataVar], Type[WeightVar]]:
    if varspec[0] in WEIGHT_SLICE_TYPES:
        return WeightVar
    else:
        return DataVar


def parse_var_class(var: str) -> Union[Type[DataVar], Type[WeightVar]]:
    if '"' in var:
        return DataVar
    else:
        return WeightVar


def is_valid_var(var: str) -> bool:
    var_class = parse_var_class(var)
    try:
        var_ = var_class.from_string(var)
        return is_valid_var_name(var_.name)
    except Exception:
        return False
