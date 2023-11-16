import inspect
from types import ModuleType
from typing import Callable


def clear_indentation(code: str) -> str:
    """Remove from all lines the common indentation, i.e.,  indentation of the first line, def function_name():
    This helps clear up the indentation caused by class definition
    """
    lines = code.split("\n")
    if len(lines) == 0:
        return ""
    indentation = len(lines[0]) - len(lines[0].lstrip())
    assert lines[0].lstrip().startswith("def "), "Not a function definition"
    return "\n".join([line[indentation:] for line in lines])


def apply_indentation(code: str, indentation: str = "    ") -> str:
    return "\n".join([indentation + line for line in code.split("\n")])


def reapply_indentation(code: str, indentation: str = "    ") -> str:
    code = clear_indentation(code)
    return apply_indentation(code, indentation)


def set_method(module_obj: ModuleType, method_name: str, method_body: str):
    # Apply indentation to method body
    exec(
        f"""
def {method_name}():
{clear_indentation(method_body)}
""",
        # module_obj is the module where the a variable to be printed is defined
        module_obj.__dict__,
    )


def set_instance_method(
    instance: ..., module_obj: ModuleType, method_name: str, method_body: str
):
    """module_obj is the module where the instance class is defined
    Based on github.com/K-Wu/cupy_playground/python/notebooks/playground/try_modify_code/try_modify_code_part_1.py
    """
    set_method(module_obj, method_name, method_body)

    from types import MethodType

    setattr(
        instance,
        method_name,
        MethodType(module_obj.__dict__[method_name], instance),
    )


def retrieve_function_definition(func: Callable) -> str:
    """Retrieve the function definition of func as a string, including the signature with default values"""

    # inspect.getsource involves the def line, i.e., signature with default values
    func_source = inspect.getsource(func)

    # TODO: get signature and default value in cases where the signature of the function is changed
    # func_signature = inspect.signature(func)
    # param_using_default_values_set = set()
    # for param in func_signature.parameters.values():
    #     if not param.default is param.empty:
    #         param_using_default_values_set.add(param.name)

    return func_source


if __name__ == "__main__":
    print(retrieve_function_definition(inspect.getmembers))
    bf = inspect.BlockFinder()

    # The indent is preserved, involving the indentation caused by class definition
    print(retrieve_function_definition(bf.__init__))
