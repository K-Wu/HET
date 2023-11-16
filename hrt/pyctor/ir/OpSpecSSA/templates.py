import inspect

# 'Wrap the string in a call to inspect.cleandoc and it will clean it up the same way docstrings get cleaned up (removing leading and trailing whitespace, and any level of common indentation).' quoted from https://stackoverflow.com/a/54429694
incode_template_dummy = inspect.cleandoc(
    """helloworld()
       helloworld()
       helloworld()"""
)
