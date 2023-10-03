# code from https://stackoverflow.com/questions/58597680/how-can-a-python-decorator-change-calls-in-decorated-function
from types import FunctionType
from functools import wraps


def reroute_decorator(**kwargs):
    def actual_decorator(func):
        globals = func.__globals__.copy()
        globals.update(kwargs)
        if "debug" in kwargs:
            print("globals:", globals)
        print(func.__code__.co_code)
        new_func = FunctionType(
            func.__code__,
            globals,
            name=func.__name__,
            argdefs=func.__defaults__,
            closure=func.__closure__,
        )
        new_func.__dict__.update(func.__dict__)

        return wraps(func)(new_func)

    return actual_decorator


import argparse
import try_args_dummy
from try_args_dummy import try_intercept2


@reroute_decorator(argparse=try_args_dummy, debug=True)
def try_intercept():
    parser = argparse.ArgumentParser()
    if parser is None:
        print("parser is None")
    else:
        print("parser is not None")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if parser is None:
        print("parser is None")
    else:
        print("parser is not None")
    try_intercept()
    reroute_decorator(A=try_args_dummy, debug=True)(
        try_args_dummy.try_intercept2
    )()
    reroute_decorator(A=try_args_dummy, debug=True)(try_intercept2)()
