import re


def modify_indent(
    py_template: str, original_indent: int, new_indent: int
) -> str:
    """Modify the indentation of a python template string. This
    function is useful when you want to change the indentation of a
    multi-line string."""
    new_py_template = ""
    for line in py_template.splitlines():
        if line.strip() == "":
            new_py_template += "\n"
        else:
            # Figure out the levels of indent
            curr_line_indent = 0
            for c in line:
                if c == " ":
                    curr_line_indent += 1
                else:
                    break
            new_curr_line_indent = int(
                curr_line_indent / original_indent * new_indent
            )
            new_py_template += (
                " " * new_curr_line_indent + line[curr_line_indent:] + "\n"
            )
    return new_py_template


def camel_case_to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def snake_case_to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.title() for word in name.split("_"))
