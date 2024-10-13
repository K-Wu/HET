#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape

# import filter app from jinja2
import enum

incode_template = """{# This jinja template is a copy of hrt/pyctor/tests/test.jinja that shows how to use jinja template to generate code. #}
res = {{ matmul_autodiff_func_name }}({{ matmul_autodiff_func_args }})
# try curly
{{ '{' }}this is interesting{{ '}' }}
{{ '{{' }}this is interesting{{ '}}' }}
{% for def in defs %}
{{ def }}
{% endfor %}
"""


class MyEnum(enum.Enum):
    A = 1
    B = 2
    C = 3

    def __eq__(self, other):
        if isinstance(other, MyEnum):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return False

    def isA(self):
        return self == MyEnum.A

    def print(self):
        return self.name


incode_template2 = """
{% if my_enum.isA() %}
    {{ "A" }}
{% elif my_enum == 2 %}
    {{ "B" }}
{% else %}
    {{ "C" }}
{% endif %}
{%- if (not False) and True -%}
    {{ "True" }}
{%- endif -%}
{{my_enum.print()}}
"""


incode_template3 = """
{% for var, type in vars %}
    {{ var }}: {{ type }}
{% endfor %}
"""

if __name__ == "__main__":
    from ..ir.OpSpecSSA.templates.environ import env

    # env = Environment(
    #     loader=FileSystemLoader(searchpath="./pyctor/"),
    #     autoescape=select_autoescape(),
    # )
    args: dict[str, str] = {"a": "a", "b": "b"}

    def try_print_file_template():
        template = env.get_template("tests/test.jinja")
        print(
            template.render(
                matmul_autodiff_func_name="test_func",
                matmul_autodiff_func_args=",".join(
                    [key + " = " + args[key] for key in args]
                ),
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        # Printed text:
        # res = test_func(a = a,b = b)

    def try_print_inline_template():
        template2 = env.from_string(incode_template)
        print(
            template2.render(
                matmul_autodiff_func_name="test_func",
                matmul_autodiff_func_args=",".join(
                    [key + " = " + args[key] for key in args]
                ),
                defs=["def1\ndef1\ndef1", "def2\ndef2\ndef2"],
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        # Printed text:
        # res = test_func(a = a,b = b)
        # # try curly
        # {this is interesting}
        # {{this is interesting}}

    def try_print_inline_template2():
        template3 = env.from_string(incode_template2)
        print(
            template3.render(
                my_enum=MyEnum.A,
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        print(
            template3.render(
                my_enum=MyEnum.B,
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        # Printed text:
        # A
        # TrueA
        # B
        # TrueB

    def try_print_inline_template3():
        template4 = env.from_string(incode_template3)
        print(
            template4.render(
                vars=[("a", "int"), ("b", "float")],
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        # Printed text:
        # a: int
        # b: float

    def test_torch_export_jinja():
        template = env.get_template(
            "ir/OpSpecSSA/templates/torch_export.inc.h.jinja"
        )
        print(
            template.render(
                func_python_name_to_cpp_name={
                    "test_func": "test_func_launcher"
                },
            ).strip()  # Remove newline trailing and ahead caused by comments
        )
        # Printed text:

    def test_cleandoc():
        import inspect

        # 'Wrap the string in a call to inspect.cleandoc and it will clean it up the same way docstrings get cleaned up (removing leading and trailing whitespace, and any level of common indentation).' quoted from https://stackoverflow.com/a/54429694
        incode_template_dummy = inspect.cleandoc(
            """helloworld()
            helloworld()
            helloworld()"""
        )

    test_cleandoc()
    try_print_file_template()
    try_print_inline_template()
    try_print_inline_template2()
    try_print_inline_template3()
    # test_torch_export_jinja()
