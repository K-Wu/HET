#!/usr/bin/env python3
import inspect
from functools import wraps
import traceback

# usage example:
# @warn_default_arguments
# def foo(a, b=1):
#     pass
def warn_default_arguments(f):
    varnames = inspect.getfullargspec(f)[0]

    @wraps(f)
    def wrapper(*a, **kw):
        explicit_params_set = set(list(varnames[: len(a)]) + list(kw.keys()))
        param_using_default_values_set = set()
        for param in inspect.signature(f).parameters.values():
            if not param.default is param.empty:
                if param.name not in explicit_params_set:
                    param_using_default_values_set.add(param.name)
        if len(param_using_default_values_set) > 0:
            print(
                "WARNING: When calling {}, the following parameters are using default values: {}".format(
                    f.__qualname__, param_using_default_values_set
                )
            )
            INDENT = 4 * " "
            callstack = "------->|" + "\n".join(
                [INDENT + line.strip() for line in traceback.format_stack()][:-1]
            )
            print("------->|{}() called:".format(f.__name__))
            print(callstack)
        return f(*a, **kw)

    return wrapper
