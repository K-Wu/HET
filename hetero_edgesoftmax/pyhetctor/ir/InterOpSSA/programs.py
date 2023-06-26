#!/usr/bin/env python3
from .variables import VarBase, is_valid_var_name, DataVar, WeightVar
from .operators import OpBase, func_name_to_op
from typing import Union
from . import program_serializer
import re


def strip_white_spaces(line: str) -> str:
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


class VariableTable:
    """
    this serves to store variable information in a program, including shape,
    occurrences of the variable the calculation is done at the first time and
    stored in the table
    """

    vars: set[str]
    vars_shape: dict[str, str]

    def __init__(self, vars_shape: Union[dict[str, str], None] = None):
        """create a variable table from a shape table or from scratch"""
        if vars_shape is not None:
            # creation from a shape table
            self.vars_shape = vars_shape
            self.vars = set(vars_shape.keys())
        else:
            # creation from scratch
            self.vars = set()
            self.vars_shape = dict()

    @classmethod
    def loads(cls, lines: list[str]) -> "VariableTable":
        """
        initiate the variable table by loading the shape info in the text
        :param lines begin with "ShapeTable{", and end with "}". To read a file, specify this parameter as fd.readlines()
        For now, we assume nothing else left in the first and the last line
        """
        assert strip_white_spaces(lines[0].strip()) == "ShapeTable{"
        assert strip_white_spaces(lines[-1].strip()) == "}"
        vars_shape: dict[str, str] = dict()
        for line in lines[1:-1]:
            line = line.strip()
            if len(line) == 0:
                continue
            var, shape = line.split(":")
            var = var.strip()
            shape = shape.strip()
            vars_shape[var] = shape
        return cls(vars_shape)

    def dumps(self) -> str:
        """output the variable table in the text, i.e., the shape table"""
        result = "ShapeTable{\n"
        for var, shape in self.vars_shape.items():
            result += f"{var}: {shape}"
            result += "\n"
        result += "}"
        return result

    # TODO: implement shape in this table
    def get_shape(self, var: Union[DataVar, WeightVar]):
        """get the shape of a variable"""
        if var.name not in self.vars_shape:
            raise ValueError(
                f"Variable {var.name} not found in the table. please run infer_shapes() first"
            )
        return self.vars_shape[var.name]

    def get_temp_var(self, hint: VarBase) -> VarBase:
        """
        This method creates and returns a new variable. The variable type, i.e.,
        weight or data, and slice_type/type will be the same as the hint
        And the variable name will be based on it as well.
        """

        def rreplace(s, old, new, occurrence):
            """
            replace the last occurrence of old with new
            source: https://stackoverflow.com/a/2556252
            """
            li = s.rsplit(old, occurrence)
            return new.join(li)

        new_temp_var: dict["str", "str"] = hint.to_dict()

        if new_temp_var["name"].find("_tmp") != -1:
            tmp_values = re.findall(r"(?<=tmp)\d+", new_temp_var["name"])

            if new_temp_var["name"].rfind("_tmp") > new_temp_var["name"].find("_delta"):
                assert len(tmp_values) == 2
            else:
                assert len(tmp_values) == 1

            # assign a new tmp number to _tmp substring
            curr_tmp_value = int(tmp_values[-1])
            while 1:
                curr_tmp_value += 1
                new_temp_var["name"] = rreplace(
                    new_temp_var["name"],
                    "_tmp" + tmp_values[-1],
                    "_tmp" + str(curr_tmp_value),
                    1,
                )
                if new_temp_var["name"] not in self.vars:
                    break
        else:
            new_temp_var["name"] += "_tmp1"

        # create a new variable
        new_var = hint.from_dict(new_temp_var)
        # TODO: get shape
        self.vars.add(new_var.name)
        return new_var


class Program:
    operations: list[OpBase]
    var_table: VariableTable

    def __init__(self, var_table: VariableTable, operations: list[OpBase]):
        self.var_table = var_table
        self.operations = operations

    # TODO: get use-def chain
    def get_users_of_result(self, operation: OpBase) -> list[OpBase]:
        raise NotImplementedError

    def get_defining_op(self, var: str) -> OpBase:
        raise NotImplementedError

    def get_seqid(self, op: OpBase) -> int:
        """returns the sequence id of the operation"""
        assert op in self.operations
        return self.operations.index(op)

    def validate(self) -> None:
        # returns True if 1) every operation has all key-value pairs correctly
        # defined as specified in this file, and 2) use-def chain is correct
        for var in self.var_table.vars:
            assert self.get_defining_op(var) is not None
            assert is_valid_var_name(var)
        for op in self.operations:
            op.validate()
            for operand in op.get_operands():
                assert operand in self.var_table.vars
                # operand should either be a weight, or defined before
                if not isinstance(operand, WeightVar):
                    assert self.get_seqid(
                        self.get_defining_op(operand)
                    ) < self.get_seqid(op)
        return

    def dumps(self) -> str:
        """Returns a multi-line string that contains the variable shape table,
        and operations. In other words, the result is in the format of
        .inter-op-ssa file"""
        result = ""
        result += self.var_table.dumps()
        result + "\nDAG{"
        for op in self.operations:
            result += op.to_string()
            result + "\n"
        result += "}\n"
        return result

    @classmethod
    def loads(cls, lines: list[str]) -> "Program":
        var_table, op_strs = program_serializer.program_loads(lines)
        ops: list[OpBase] = []
        for op in op_strs:
            ops.append(func_name_to_op[op["func_name"]].from_keyval_pairs(op))

        return cls(var_table, ops)

    def differentiate(self) -> "Program":
        """
        differentiate the program, and return the differentiated program
        """
        diff_var_table = VariableTable()
        diff_ops: list[OpBase] = []
        for op in self.operations:
            diff_ops += op.differentiate()
        for op in diff_ops:
            for result in op.get_results():
                diff_var_table.vars.add(result.to_string())
        return Program(diff_var_table, diff_ops)

    def infer_shapes():
        """
        after differentiation or pattern match from the inter-op IR, we get all
        operations and unique variable names. We need to infer the shapes of all
        variables.
        """
        raise NotImplementedError
