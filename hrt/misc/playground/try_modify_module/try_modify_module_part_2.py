"""This script will work and produce the same results even if the three parts are not organized as a package/subpackge, i.e., in a folder with __init__.py and executed by python -m try_modify_module.try_modify_module_part_2"""

if __name__ == "__main__":
    from . import try_modify_module_part_0
    from .try_modify_module_part_1 import set_print_a_from_part_0

    set_print_a_from_part_0()
    try_modify_module_part_0.print_a()

    # The following shows class method can be modified in a similar way
    from .try_modify_module_part_0 import PrintA
    from .try_modify_module_part_1 import set_print_a_class_method_from_part_0

    a = PrintA("test string of bonus_b")

    # This prints 15 because the class method is not modified yet
    a.print_a()
    set_print_a_class_method_from_part_0(a)
    # This prints 20 because the class method is modified
    a.print_a()
