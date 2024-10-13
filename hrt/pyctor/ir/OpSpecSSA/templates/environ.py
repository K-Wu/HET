import sys
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os

# Import hrt/utils/detect_pwd.py
from utils.detect_pwd import is_pwd_het_dev_root

# This is a pointer to the module object instance itself.
# Reference: https://stackoverflow.com/a/35904211/5555077
this = sys.modules[__name__]
this.initialized = False
this.env: Environment
# this.templates_asfile: dict[str, Template] = dict()


def initialize():
    assert is_pwd_het_dev_root(), (
        "Please run this script at the /hrt directory so that the jinja"
        " templates can be found."
    )
    this.env = Environment(
        loader=FileSystemLoader(searchpath="./pyctor/"),
        autoescape=select_autoescape(),
    )
    this.initialized = True
    # this.templates_asfile = dict()
    # for root, dirs, files in os.walk("./pyctor/"):
    #     for file in files:
    #         if file.endswith(".jinja"):
    #             template = this.env.get_template(file)
    #             this.templates_asfile[file] = template


if not this.initialized:
    initialize()
