#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape

if __name__ == "__main__":
    env = Environment(
        loader=FileSystemLoader(searchpath="./pyhetctor/tests/"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("test.jinja")
    args = {"a": "a", "b": "b"}
    print(
        template.render(
            matmul_autodiff_func_name="test_func",
            matmul_autodiff_func_args=",".join(
                [key + " = " + args[key] for key in args]
            ),
        )
    )
