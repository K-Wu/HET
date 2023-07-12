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
                [
                    "------->|" + INDENT + line.strip()
                    for line in traceback.format_stack()
                ][:-1]
            )
            print("------->|{}() called:".format(f.__name__))
            print(callstack)
        return f(*a, **kw)

    return wrapper


# code from https://stackoverflow.com/questions/58597680/how-can-a-python-decorator-change-calls-in-decorated-function
from types import FunctionType


# usage example:
# @reroute_namespace(argparse=test_args_dummy)
# def try_intercept():
#     parser = argparse.ArgumentParser()
#     if parser is None:
#         print("parser is None")
#     else:
#         print("parser is not None")
def reroute_namespace(**kwargs):
    def actual_decorator(func):
        globals = func.__globals__.copy()
        globals.update(kwargs)
        if "debug" in kwargs:
            print("globals:", globals)
        new_func = FunctionType(
            func.__code__,
            globals,
            name=func.__name__,
            argdefs=func.__defaults__,
            closure=func.__closure__,
        )
        new_func.__dict__.update(func.__dict__)
        new_func = wraps(func)(
            new_func
        )  # this is necessary to preserve the original function name
        return new_func

    return actual_decorator


# specify the other function as a keyword argument, e.g.,
# @assert_signature_equal_decorator(other_func=foo)
# def this_func(a, b):
#    pass
def assert_signature_equal_to(**kwargs):
    def actual_decorator(func):
        other_func = kwargs["other_func"]
        assert inspect.getfullargspec(func) == inspect.getfullargspec(other_func)
        return func

    return actual_decorator
