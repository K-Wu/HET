#!/usr/bin/env python3
# from https://stackoverflow.com/a/55902915/5555077

# from contextlib import contextmanager
#
# @contextmanager
# def nullcontext(enter_result=None):
#     yield enter_result

import contextlib

cm = contextlib.nullcontext()

if __name__ == "__main__":
    with cm as context:
        print("abc")
