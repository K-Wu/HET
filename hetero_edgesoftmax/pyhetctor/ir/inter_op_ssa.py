#!/usr/bin/env python3


class LinearOp:
    def __init__(self):
        pass

    def load_from(op_dict):
        raise NotImplementedError

    def dump_to(op_dict):
        raise NotImplementedError

    def validate():
        raise NotImplementedError


# returns True if 1) every instruction has all key-value pairs correctly defined as specified in this file, and 2) use-def chain is correct
def validate():
    raise NotImplementedError
