#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader(searchpath="./pyctor/ir/OpSpecSSA/templates/"),
    autoescape=select_autoescape(),
)


def codgen_main_procedure():
    raise NotImplementedError
