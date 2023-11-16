#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape

incode_template = """{# This jinja template is a copy of hrt/pyctor/tests/test.jinja that shows how to use jinja template to generate code. #}
res = {{ matmul_autodiff_func_name }}({{ matmul_autodiff_func_args }})
# try curly
{{ '{' }}this is interesting{{ '}' }}
{{ '{{' }}this is interesting{{ '}}' }}
"""

if __name__ == "__main__":
    env = Environment(
        loader=FileSystemLoader(searchpath="./pyctor/tests/"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("test.jinja")
    args: dict[str, str] = {"a": "a", "b": "b"}
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

    template2 = env.from_string(incode_template)
    print(
        template2.render(
            matmul_autodiff_func_name="test_func",
            matmul_autodiff_func_args=",".join(
                [key + " = " + args[key] for key in args]
            ),
        ).strip()  # Remove newline trailing and ahead caused by comments
    )
    # Printed text:
    # res = test_func(a = a,b = b)
    # # try curly
    # {this is interesting}
