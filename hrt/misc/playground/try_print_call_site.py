# From https://stackoverflow.com/a/71228259 and https://stackoverflow.com/a/64427884/5555077 and https://stackoverflow.com/a/58166804/5555077 and https://stackoverflow.com/a/48653175/5555077
import os
from inspect import getframeinfo, stack
import traceback
from functools import wraps
from recordclass import dataobject
from typing import Callable


class CallRecord(dataobject):
    callstack: list[str]
    funcname: str


def log_and_call(statement):
    def decorator(func):
        caller = getframeinfo(stack()[1][0])

        @wraps(func)
        def wrapper(*args, **kwargs):
            # set name_override to func.__name__
            print(
                statement,
                "name_override",
                func.__name__,
                "file_override",
                os.path.basename(caller.filename),
                "line_override",
                caller.lineno,
            )
            INDENT = 4 * " "
            callstack_multilines: list[list[str]] = [
                line.strip().split("\n") for line in traceback.format_stack()
            ][:-1]
            callstack_lines = [
                INDENT + line
                for multiline in callstack_multilines
                for line in multiline
            ]
            callstack = "------->|" + "\n------->|".join(callstack_lines)
            print(
                CallRecord(
                    callstack=traceback.format_stack(), funcname=func.__name__
                )
            )
            print("------->|{}() called:".format(func.__name__))
            print(callstack)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MyClass:
    dummy_var: list[Callable] = list()

    @log_and_call("This should be logged by 'decorated_function'")
    def decorated_function(
        self,
    ):  # <- the logging in the wrapped function will point to/log this line for lineno.
        print("I ran")


if __name__ == "__main__":
    # ❯ python misc/playground/try_print_call_site.py
    # This should be logged by 'decorated_function' name_override decorated_function file_override try_print_call_site.py line_override 40
    # ------->|decorated_function() called:
    # ------->|    File "/home/kwu/HET/hrt/misc/playground/try_print_call_site.py", line 45, in <module>
    #     decorated_function()
    # I ran

    mc = MyClass()
    mc.decorated_function()
