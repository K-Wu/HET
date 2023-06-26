#!/usr/bin/env python3
from typing import NamedTuple, Type
import abc


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
    def validate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_delta(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def lower(self):
        raise NotImplementedError

    @abs.abstractmethod
    def from_dict(cls, d: dict["str", "str"]) -> "VarBase":
        raise NotImplementedError

    @abs.abstractmethod
    def to_dict(self) -> dict["str", "str"]:
        raise NotImplementedError


# TODO: do we need "shape" and "dtype"?
_WeightVar = NamedTuple("WeightVar", [("name", str), ("slice_type", str)])
_DataVar = NamedTuple("DataVar", [("type", str), ("name", str)])

WEIGHT_SLICE_TYPES = {"EDGETYPE": 0, "NODETYPE": 1, "NONE": 2}
DATA_TYPES = {"EDGEWISE": 0, "NODEWISE": 1, "DSTNODE": 2, "SRCNODE": 3}


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

    def to_dict(self) -> dict["str", "str"]:
        return {"name": self.name, "slice_type": self.slice_type}

    def to_string(self) -> str:
        return f"({self.name}, {self.slice_type})"

    def validate(self):
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
    def from_dict(cls, d: dict["str", "str"]) -> "DataVar":
        return cls(type=d["type"], name=d["name"])

    def to_dict(self) -> dict["str", "str"]:
        return {"type": self.type, "name": self.name}

    def to_string(self) -> str:
        return f'({self.type}, "{self.name}")'

    def validate(self):
        assert self.type in DATA_TYPES
        assert is_valid_var_name(self.name)

    def lower(self):
        raise NotImplementedError

    def is_delta(self) -> bool:
        return "_delta" in self.name

    def differentiate(self):
        return DataVar(type=self.type, name=self.name + "_delta")


def get_var_class(name: str) -> Type[VarBase]:
    if '"' in name:
        return DataVar
    else:
        return WeightVar
