""" 'Wrap the string in a call to inspect.cleandoc and it will clean it up the same way docstrings get cleaned up (removing leading and trailing whitespace, and any level of common indentation).' quoted from https://stackoverflow.com/a/54429694
"""

import inspect

if __name__ == "__main__":
    s = r"""jkljkj{abc}
    dfkjslfds
        sqjdlqkj"""
    print(inspect.cleandoc(s))
