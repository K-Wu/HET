#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape
from ...utils import is_pwd_het_dev_root

assert (
    is_pwd_het_dev_root()
), "Please run this script from het_dev root directory."

env = Environment(
    loader=FileSystemLoader(searchpath="./pyctor/ir/OpSpecSSA/templates/"),
    autoescape=select_autoescape(),
)


def codgen_main_procedure():
    raise NotImplementedError
