import test_modify_module_part_0


def my_print_a():
    print(a + 5)


def set_print_a(module_obj):
    exec(
        """
def print_a():
    print(a + 5)
         """,
        module_obj.__dict__,
    )


if __name__ == "__main__":
    # The following won't work because test_modify_module_part_1 does not have defined the a variable
    try:
        test_modify_module_part_0.print_a = my_print_a
        test_modify_module_part_0.print_a()
    except Exception as e:
        print(e)

    # The following will work because the module function has been updated by setting the exec namespace
    test_modify_module_part_0.set_print_a = set_print_a
    test_modify_module_part_0.set_print_a(test_modify_module_part_0)
    test_modify_module_part_0.print_a()

    # The following still won't work
    try:
        test_modify_module_part_0.print_a = my_print_a
        test_modify_module_part_0.print_a()
    except Exception as e:
        print(e)

    # The following will also work
    set_print_a(test_modify_module_part_0)
    test_modify_module_part_0.print_a()
