#!/usr/bin/env python3
from .variables import VarBase
from .operators import OpBase
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

    @classmethod
    def loads(cls, lines: list[str]) -> "VariableTable":
        """
        initiate the variable table by loading the shape info in the text
        :param lines begin with "ShapeTable{", and end with "}". To read a file, specify this parameter as fd.readlines()
        For now, we assume nothing else left in the first and the last line
        """
        assert strip_white_spaces(lines[0].strip()) == "ShapeTable{"
        assert strip_white_spaces(lines[-1].strip()) == "}"
        raise NotImplementedError

    def dumps(self) -> str:
        """output the variable table in the text, i.e., the shape table"""
        raise NotImplementedError

    # TODO: implement shape in this table
    def get_shape(self, var: VarBase):
        raise NotImplementedError

    # TODO: get use-def chain
    def get_users_of_result(self, operation: OpBase) -> list[OpBase]:
        raise NotImplementedError

    def get_defining_op(self, var: VarBase) -> OpBase:
        raise NotImplementedError

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
                if new_temp_var["name"] not in self.var_table:
                    break
        else:
            new_temp_var["name"] += "_tmp1"

        # create a new variable
        new_var = hint.from_dict(new_temp_var)
        self.var_table[new_var.name] = new_var
        return new_var


class Program:
    operations: list[OpBase]
    var_table: VariableTable
    raise NotImplementedError

    def validate(self):
        # returns True if 1) every operation has all key-value pairs correctly
        # defined as specified in this file, and 2) use-def chain is correct
        raise NotImplementedError

    def dumps(self) -> str:
        result = "ShapeTable{"
        result += self.var_table.dumps()
        result += "\n}"
        result + "\nDAG{"
        for op in self.operations:
            result += op.to_string()
            result + "\n"
        result += "}\n"
        return result
