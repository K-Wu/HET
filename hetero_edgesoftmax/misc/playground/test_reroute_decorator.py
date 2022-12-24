# code from https://stackoverflow.com/questions/58597680/how-can-a-python-decorator-change-calls-in-decorated-function
from types import FunctionType


def reroute_decorator(**kwargs):
    def actual_decorator(func):
        globals = func.__globals__.copy()
        globals.update(kwargs)
        new_func = FunctionType(
            func.__code__,
            globals,
            name=func.__name__,
            argdefs=func.__defaults__,
            closure=func.__closure__,
        )
        new_func.__dict__.update(func.__dict__)
        return new_func

    return actual_decorator


import argparse
import test_args_dummy


@reroute_decorator(argparse=test_args_dummy)
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
