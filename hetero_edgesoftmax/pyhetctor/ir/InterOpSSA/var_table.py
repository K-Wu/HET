#!/usr/bin/env python3
from .variables import *


# this serves to store variable information in a program, including shape, occurrences of the variable
# the calculation is done at the first time and stored in the table
class VariableTable:
    # initiate the variable table by loading the shape info in the text
    @classmethod
    def loads(cls, lines):
        raise NotImplementedError

    def dumps(self):
        raise NotImplementedError

    # TODO: implement shape in this table
    def get_shape(self, var):
        raise NotImplementedError

    # TODO: get use-def chain
    def get_users_of_result(self, operation):
        raise NotImplementedError

    def get_defining_op(self, var):
        raise NotImplementedError
