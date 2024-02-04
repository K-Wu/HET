from typing import (
    Generic,
    TypeVar,
)
import traceback
from functools import wraps
from recordclass import dataobject
import queue
from ordered_set import OrderedSet


class CallRecord(dataobject):
    callstack: list[str]
    funcname: str
    msg: str


# From hrt/misc/playground/try_print_call_site.py and https://stackoverflow.com/questions/60219591/using-a-paramaterized-decorator-for-recording-methods-in-a-class
def log_pass_calls(description: str):
    """Decorate class functions that do analysis or transform pass and record the call site in the called_function list of the class instance"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = ("STARTED | FUNCTION: {} | ARGS: {} | KWARGS: {} ").format(
                func.__name__, args, kwargs
            )
            # print(msg)

            args[0].passes_call_records.append(
                CallRecord(
                    callstack=traceback.format_stack(),
                    funcname=func.__name__,
                    msg=msg,
                )
            )  # i.e., self.called_function

            return func(*args, **kwargs)

        return wrapper

    return decorator


T = TypeVar("T")


class MySet(set[T], Generic[T]):
    """
    Set that records analysis passes and transform passes.
    Example:
    ```
    class Program:
        analysis_passes: MySet[Callable]
        transform_passes: MySet[Callable]

        @transform_passes.register
        def do_something(self):
            ...

        @analysis_passes.register
        def check_something(self):
            ...
    ```
    From https://stackoverflow.com/questions/50372342/class-with-a-registry-of-methods-based-on-decorators
    """

    def register(self, method):
        self.add(method)
        return method


class OrderedSetQueue(queue.Queue):
    # From https://stackoverflow.com/a/16506527/5555077
    def _init(self, maxsize):
        self.queue = OrderedSet()

    def _put(self, item):
        current_length = len(self.queue)
        self.queue.add(item)

        # If the item is already in the queue, remove and readd it so that it is moved to the end, i.e., the next item to be popped
        if current_length == len(self.queue):
            self.move_to_end(item)

    def move_to_end(self, item):
        assert item in self.queue
        self.queue.pop(self.queue.index(item))
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()

    def __contains__(self, item):
        with self.mutex:
            return item in self.queue

    def _qsize(self):
        return len(self.queue)
