#!/usr/bin/env python3
from collections import namedtuple

# TODO: do we need "shape" and "dtype"?
_WeightVar = namedtuple("WeightVar", ["name", "slice_type"])
_DataVar = namedtuple("DataVar", ["type", "name"])


_LinearOp = namedtuple("LinearOp", ["result", "left", "right"])


# when inherited, this class provides implementation of from_dict for namedtuples
class DictLoader:
    @classmethod
    def from_dict(cls, d):
        # return cls(**d)
        return cls._make(d)

    def to_dict(self):
        return self._asdict()


class LinearOp(_LinearOp, DictLoader):
    def validate():
        raise NotImplementedError


class WeightVar(_WeightVar, DictLoader):
    def validate():
        raise NotImplementedError


class DataVar(_DataVar, DictLoader):
    def validate():
        raise NotImplementedError


# returns True if 1) every instruction has all key-value pairs correctly defined as specified in this file, and 2) use-def chain is correct
def validate():
    raise NotImplementedError
