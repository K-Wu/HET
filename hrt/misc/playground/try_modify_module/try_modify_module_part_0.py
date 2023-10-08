"""This script will work and produce the same results even if the three parts are not organized as a package/subpackge, i.e., in a folder with __init__.py and executed by python -m try_modify_module.try_modify_module_part_0"""

a = 15


def print_a():
    print(a)


class PrintA:
    """The class version if print_a to demonstrate how to modify a class method in a module"""

    def __init__(self, bonus_b):
        # store bonus_b to demonstrate in case the class method is modified, if the modified method could retrieve the bonus_b associated with the instance
        self.bonus_b = bonus_b

    def print_a(self):
        print(a)
