from .variables import (
    VarBase,
    DataVar,
    WeightVar,
    parse_var_class,
    Shape,
)
from .operators import OpBase, FusedOpBase
from typing import (
    Union,
    NamedTuple,
    Annotated,
)

import re

DefUseEntry = NamedTuple(
    "DefUseEntry",
    [
        ("name", str),
        ("def_op", Union[OpBase, None]),
        ("use_ops", list[OpBase]),
    ],
)


def remove_white_spaces(line: str) -> str:
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


class VariableTable:
    """
    this serves to store variable information in a program, including shape,
    occurrences of the variable the calculation is done at the first time and
    stored in the table

    Serial Format:
    each entry is in the format of
    shape: var_key <- var_name2 <- var_name3 <- ...
    where shape is one of (matrix, vector, scalar) for data var, and
    (none, nodetype, edgetype) + shape per type (matrix, vector, scalar) in the
    case of weight var
    var_key involves both name and (slice_)type;
    var_name2, var_name3 are the different value number of the same variable,
    and only stores their name with (slice_)type omitted
    """

    dsl_vars: Annotated[
        set[VarBase],
        "variables defined during the lowering from Inter Op DSL to SSA",
    ]

    ##
    ## ShapeInferer
    vars_shape: dict[str, Shape]

    ##
    ## ValueNumberer
    vars_input: set[VarBase]
    numbered_val_to_key: Annotated[
        dict[VarBase, VarBase],
        """map the full string representation of numbered variable to full string representation of variable in its original name, e.g., (EDGEWISE, "var_name2") to (EDGEWISE, "var_name")""",
    ]
    numbered_key_vals: Annotated[
        dict[str, list[VarBase]],
        """
    reverse of numbered_val_to_key""",
    ]

    ##
    ## DefUseAnalyzer
    def_use_table: Annotated[
        # after SSA numbering, each value will be a single DefUseEntry
        dict[str, list[DefUseEntry]],
        """
    numbered_key_vals and def_use_table together store the value numbering information

    numbered_key_vals
    before value numbering, each (key, value) in numbered_key_vals looks like
    (var_key, [var_name]) (another_var, [another_var]) ...
    after value numbering, the above entry may now look like
    (var_key, [var_name, var_name2, var_name3]) (another_var, [another_var])

    def_use_table
    whether before or after value numbering, each (key, value) in def_use_table looks like
    (var_key, [DefUseEntry(var_name, opid0, [opid1]), DefUseEntry(var_name(2), opid2, [opid3, opid4])])
    (another_var, [DefUseEntry(another_var, opid5, [opid6])])
    """,
    ]

    def __init__(self, var_table: Union["VariableTable", None] = None):
        """create a variable table from a shape table or from scratch"""
        if var_table is not None:
            # shallow copies
            self.vars_shape = var_table.vars_shape.copy()
            self.dsl_vars = var_table.dsl_vars.copy()
            self.vars_input = var_table.vars_input.copy()
            self.numbered_key_vals = var_table.numbered_key_vals.copy()
            self.numbered_val_to_key = var_table.numbered_val_to_key.copy()
            self.def_use_table = var_table.def_use_table.copy()
        else:
            # creation from scratch
            self.vars_shape = dict()
            self.dsl_vars = set()
            self.vars_input = set()
            self.numbered_key_vals = dict()
            self.numbered_val_to_key = dict()
            self.def_use_table = dict()

    def get_var_key(self, var: VarBase) -> VarBase:
        """
        This method returns the var_key corrsponding to the param var to be used
        to query shape information from self.vars_shape
        """
        # There are two scenarios, either it is after value numbering or not.
        # After value numbering, var is a value number from operation after numbering, and self.numbered_val_to_key is produced
        # Before value numbering, var is itself the var key
        if var in self.numbered_val_to_key:
            return self.numbered_val_to_key[var]
        else:
            print("Warning: value number not done before get_var_key")
            return var

    def get_var_key_str(
        self, var: VarBase, after_value_numbering: bool | None = None
    ) -> str:
        """
        This method returns the key of the variable. The key is used to query
        shape information from self.vars_shape
        """
        if (
            after_value_numbering is not None
            and not var in self.numbered_val_to_key
        ):
            assert (
                not after_value_numbering
            ), f"Value numbering is not done for {var}"
        var_key_str = self.get_var_key(var).get_name()
        return var_key_str

    @classmethod
    def loads(cls, lines: list[str]) -> "VariableTable":
        """
        initiate the variable table by loading the shape info in the text
        :param lines begin with "VariableTable{", and end with "}". To read a file,
        specify this parameter as fd.readlines()
        For now, we assume nothing else left in the first and the last line.
        Adapted from loads_op from hrt/pyctor/ir/InterOpSSA/serialize_program.py
        """
        var_table = cls()

        assert remove_white_spaces(lines[0].strip()) == "VariableTable{"
        assert lines[-1].strip() == "}"
        lines = lines[1:-1]

        # load initial variables and weights

        from . import program_serializer

        scopes = program_serializer.find_first_level_scopes(lines)
        for scope_beg, scope_end, scope_tag in scopes:
            # For simplicity of parsing, we assume the scope beginning line only contains tag and "{"
            assert (
                remove_white_spaces(lines[scope_beg].strip())
                == scope_tag + "{"
            )
            # Similarly, we assume the scope ending line only contains "}"
            assert lines[scope_end].strip() == "}"
            if scope_tag == "InitialVariablesAndWeights":
                # Assume only one line in current scheme
                assert scope_end - scope_beg - 1 == 1
                for line in lines[scope_beg + 1 : scope_end]:
                    # Split by ';' because variable string (name, type) contains ','
                    line_var_strs = line.split(";")
                    # Remove the beginning whitespaces
                    line_var_strs = [
                        var_str.strip() for var_str in line_var_strs
                    ]
                    line_vars: set[VarBase] = {
                        parse_var_class(var_str).from_string(var_str)
                        for var_str in line_var_strs
                    }
                    var_table.vars_input.update(line_vars)  # in place update

            elif scope_tag == "VariableNumbersAndShapes":
                # load shape info, each line stores a variable's shape and all its numbered values, i.e.,
                #  shape: var_key <- var_name2 <- var_name3 <- ...
                for line in lines[scope_beg + 1 : scope_end]:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    shape, vars_str = line.split(":")
                    shape = shape.strip()
                    var_varnames_strs = vars_str.split("<-")
                    var_varnames_strs = [
                        var_str.strip() for var_str in var_varnames_strs
                    ]
                    var = parse_var_class(var_varnames_strs[0]).from_string(
                        var_varnames_strs[0]
                    )
                    varnames = var_varnames_strs[1:]
                    var_table.vars_shape[
                        var_table.get_var_key_str(var)
                    ] = Shape(type=shape)

                    var_table.numbered_key_vals[var] = [var]
                    var_table.numbered_val_to_key[var] = var
                    for varname in varnames:
                        var_newer = var.get_numbered_var(varname)
                        var_table.numbered_key_vals[var].append(var_newer)
                        var_table.numbered_val_to_key[var_newer] = var
                    var_table.vars_shape[
                        var_table.get_var_key_str(var)
                    ] = Shape(type=shape)
        return cls(var_table)

    def dumps(self) -> str:
        """output the variable table in the text, i.e., the shape table"""
        result = "VariableTable{\n"
        # Step 1: Output initial variables and weights
        result += "InitialVariablesAndWeights{\n"
        variable_strings = [var.to_string() for var in self.vars_input]
        for var_str in variable_strings:
            assert ";" not in var_str, f"Variable name {var_str} contains ';'"
        result += "; ".join([var.to_string() for var in self.vars_input])
        result += "\n}\n"

        # Step 2: Output shape info
        result += "VariableNumbersAndShapes{\n"
        for var_str, shape in self.vars_shape.items():
            result += f"{shape.type}: {var_str}"
            for var_newer in self.numbered_key_vals[var_str][1:]:
                result += f" <- {var_newer.to_string()}"
            result += "\n"
        result += "}\n"

        result += "}"

        return result

    def get_shape_info(self, var: VarBase) -> Union[Shape, None]:
        """get the shape of a variable"""
        key: str = self.get_var_key_str(var)
        if key not in self.vars_shape:
            return None
        return self.vars_shape[key]

    def get_shape_info_or_throw(self, var: Union[DataVar, WeightVar]) -> Shape:
        """get the shape of a variable"""
        result = self.get_shape_info(var)
        if result is None:
            raise ValueError(
                f"Variable {var.get_name()} not found in the table. please run"
                " infer_shapes() first"
            )
        return result

    def set_shape_info_or_throw(self, var: VarBase, new_shape_info) -> None:
        """set the shape of a variable"""
        key = self.get_var_key_str(var)
        if key in self.vars_shape and self.vars_shape[key] != new_shape_info:
            raise ValueError(
                f"Variable {var.get_name()} already has a shape"
                f" {self.vars_shape[key]}, cannot set to {new_shape_info}"
            )
        self.vars_shape[key] = new_shape_info

    def contains(self, var: VarBase) -> bool:
        """check if the variable is in the table"""
        return var in self.numbered_val_to_key

    def _create_var_by(self, hint: VarBase, suffix: str) -> VarBase:
        """
        This internal method creates and returns a new variable. The variable
        type, i.e., weight or data, and slice_type/type will be the same as the
        hint. And the variable name will be based on the hint and suffix.
        """

        def rreplace(s, old, new, occurrence):
            """
            replace the last occurrence of old with new
            source: https://stackoverflow.com/a/2556252
            """
            li = s.rsplit(old, occurrence)
            return new.join(li)

        new_temp_var: dict["str", "str"] = hint.to_dict()

        if new_temp_var["name"].find("_" + suffix) != -1:
            tmp_values = re.findall(
                r"(?<={suffix})\d+".format(suffix=suffix), new_temp_var["name"]
            )

            if new_temp_var["name"].rfind("_" + suffix) > new_temp_var[
                "name"
            ].find("_delta"):
                assert len(tmp_values) == 2
            else:
                assert len(tmp_values) == 1

            # assign a new tmp number to _tmp substring
            curr_tmp_value = int(tmp_values[-1])
            while 1:
                curr_tmp_value += 1
                new_temp_var["name"] = rreplace(
                    new_temp_var["name"],
                    "_" + suffix + tmp_values[-1],
                    "_" + suffix + str(curr_tmp_value),
                    1,
                )
                if (
                    hint.__class__.from_dict(new_temp_var).to_string()
                    not in self.dsl_vars
                ):
                    break
        else:
            new_temp_var["name"] += "_{suffix}1".format(suffix=suffix)

        # Create a new variable
        new_var = hint.__class__.from_dict(new_temp_var)
        return new_var

    def _create_temp_var(self, hint: VarBase) -> VarBase:
        return self._create_var_by(hint, "tmp")

    # This function seems unncessary
    # def _get_var_decollision(self, hint: VarBase) -> VarBase:
    #     return self._create_var_by(hint, "decollision")

    def create_temp_var_dsl(self, hint: VarBase) -> VarBase:
        """
        This method creates and returns a new variable, and add it to the table.
        It can be used during the pattern matching process that lowers Inter Op
        DSL to Inter Op SSA.
        At that time only variables names are registered in the variable table,
        and all the rest of the information in the variable table are not
        produced yet.
        Solely used by pattern_matcher.py
        """
        new_var = self._create_temp_var(hint)
        self.register_dsl_var(new_var)
        return new_var

    def register_dsl_var(self, var: VarBase) -> None:
        """
        This method registers a variable key. This is done to every op result,
        i.e., def op result, during the lowering from Inter Op DSL to Inter Op
        SSA in order to have knowledge about the existing variable names. This
        is necessary to make sure during creating temporary variable names, no
        name collision happens.
        Solely used by pattern_matcher.py
        """
        # self.numbered_val_to_name[var.to_string()] = var.to_string()
        # self.numbered_name_vals[var.to_string()] = [var.to_string()]
        self.dsl_vars.add(var)

    def differentiate(
        self, diff_ops: list[Union[OpBase, FusedOpBase]]
    ) -> "VariableTable":
        """This is used by auto_differer.py to differentiate the program"""
        raise NotImplementedError
