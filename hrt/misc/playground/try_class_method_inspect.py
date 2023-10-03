#!/usr/bin/env python3
# from https://stackoverflow.com/a/64427884/5555077 and https://stackoverflow.com/a/58166804/5555077 and https://stackoverflow.com/a/48653175/5555077
import inspect
from functools import wraps
import traceback


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
                "WARNING: When calling {}, the following parameters are using"
                " default values: {}".format(
                    f.__qualname__, param_using_default_values_set
                )
            )
            INDENT = 4 * " "
            callstack = "------->|" + "\n".join(
                [INDENT + line.strip() for line in traceback.format_stack()][
                    :-1
                ]
            )
            print("------->|{}() called:".format(f.__name__))
            print(callstack)
        return f(*a, **kw)

    return wrapper


@warn_default_arguments
def foo(a, b=1):
    pass


class MyFoo:
    @warn_default_arguments
    def __init__(self, a, b=1):
        pass


if __name__ == "__main__":
    foo(1)
    myfoo = MyFoo(1)
