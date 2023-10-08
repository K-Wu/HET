"""This script will work and produce the same results even if the three parts are not organized as a package/subpackge, i.e., in a folder with __init__.py and executed by python -m try_modify_module.try_modify_module_part_1"""
from . import try_modify_module_part_0
from types import ModuleType


def my_print_a():
    """This is a function used by the first two not-working methods in __main__"""
    print(a + 5)


def set_print_a(module_obj: ModuleType):
    exec(
        """
def print_a():
    print(a + 5)
         """,
        module_obj.__dict__,
    )


def set_print_a_from_part_0():
    """This function is used by try_modify_module_part_2.py and is working"""
    set_print_a(try_modify_module_part_0)


def set_print_a_class_method(printa_obj: ..., module_obj: ModuleType):
    # This is executed locally, rather than in the module namespace where class of printa_obj is defined
    exec(
        """
def print_a_as_class_method(self):
    print(a + 5)
    print("bonus print: retrieving instance's bonus_b: ", self.bonus_b)
""",
        # module_obj is the module where the a variable to be printed is defined
        module_obj.__dict__,
    )
    from types import MethodType

    printa_obj.print_a = MethodType(
        module_obj.__dict__["print_a_as_class_method"], printa_obj
    )


def set_print_a_class_method_from_part_0(printa_obj: ...):
    """This function is used by try_modify_module_part_2.py and is working"""
    set_print_a_class_method(printa_obj, try_modify_module_part_0)


if __name__ == "__main__":
    # The following won't work because try_modify_module_part_1 does not have defined the a variable
    try:
        try_modify_module_part_0.print_a = my_print_a
        try_modify_module_part_0.print_a()
    except Exception as e:
        print(e)

    # The following still won't work
    try:
        try_modify_module_part_0.print_a = my_print_a
        try_modify_module_part_0.print_a()
    except Exception as e:
        print(e)

    # The following will work
    set_print_a(try_modify_module_part_0)
    try_modify_module_part_0.print_a()

    # The following will also work because the module function has been updated by setting the exec namespace
    try_modify_module_part_0.set_print_a = set_print_a
    try_modify_module_part_0.set_print_a(try_modify_module_part_0)
    try_modify_module_part_0.print_a()

    # The following shows class method can be modified in a similar way
    from .try_modify_module_part_0 import PrintA

    printa = PrintA("test string of bonus_b")

    # This prints 15 because the class method is not modified yet
    printa.print_a()
    set_print_a_class_method(printa, try_modify_module_part_0)
    # This prints 20 because the class method is modified
    printa.print_a()
