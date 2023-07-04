#!/usr/bin/env python3
from .variables import VarBase, is_valid_var_name, DataVar, WeightVar
from .operators import OpBase, FusedOpBase, func_name_to_op
from typing import Union, NamedTuple, Annotated
from . import program_serializer
import re


def strip_white_spaces(line: str) -> str:
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


DefUseEntry = NamedTuple(
    "DefUseEntry", [("name", str), ("def_op_idx", int), ("use_op_idx", list[int])]
)


class VariableTable:
    """
    this serves to store variable information in a program, including shape,
    occurrences of the variable the calculation is done at the first time and
    stored in the table

    Serial Format:
    each entry is in the format of
    shape: var_name <- var_name2 <- var_name3 <- ...
    where shape is one of (matrix, vector, scalar) for data var, and
    (none, nodetype, edgetype) + shape per type (matrix, vector, scalar) in the
    case of weight var
    var_name involves both name and (slice_)type;
    var_name2, var_name3 are the different value number of the same variable,
    and only stores their name with (slice_)type omitted
    """

    vars_shape: dict[str, str]
    numbered_val_to_name: Annotated[
        dict[str, str],
        """map the full string representation to full string representation, e.g., (EDGEWISE, "var_name2") to (EDGEWISE, "var_name")""",
    ]
    numbered_name_vals: Annotated[
        dict[str, list[str]],
        """
    reverse of numbered_val_to_name""",
    ]
    def_use_table: Annotated[
        # after SSA numbering, each value will be a single DefUseEntry
        Union[dict[str, DefUseEntry], dict[str, list[DefUseEntry]]],
        """
    numbered_val_table and def_use_table together store the value numbering information
    before value numbering, each (key, value) in numbered_val_table looks like
    (var_name, [var_name]) (var2_name, [var2_name]) ...
    after value numbering, the above entry may now look like
    (var_name, [var_name, var_name2, var_name3]) (var2_name, [var2_name])

    before value numbering, each (key, value) in def_use_table looks like
    (var_name, [DefUseEntry(var_name, opid0, [opid1]), DefUseEntry(var_name, opid2, [opid3, opid4])])
    after value numbering, the above entry may now look like
    (var_name, [DefUseEntry(var_name, opid0, opid1)]),
    (var_name2, [DefUseEntry(var_name2, opid2, [opid3, opid4])])
    """,
    ]

    def __init__(self, vars_shape: Union[dict[str, str], None] = None):
        """create a variable table from a shape table or from scratch"""
        if vars_shape is not None:
            # creation from a shape table
            self.vars_shape = vars_shape
            self.numbered_val_to_name = {k: k for k in vars_shape.keys()}
        else:
            # creation from scratch
            self.numbered_val_to_name = dict()
            self.vars_shape = dict()

    # TODO: update to reflect the new serialize scheme
    @classmethod
    def loads(cls, lines: list[str]) -> "VariableTable":
        """
        initiate the variable table by loading the shape info in the text
        :param lines begin with "ShapeTable{", and end with "}". To read a file,
        specify this parameter as fd.readlines()
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

    # TODO: update to reflect the new serialize scheme
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

    def _get_var_by(self, hint: VarBase, suffix: str) -> VarBase:
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

            if new_temp_var["name"].rfind("_" + suffix) > new_temp_var["name"].find(
                "_delta"
            ):
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
                    hint.from_dict(new_temp_var).to_string()
                    not in self.numbered_val_to_name
                ):
                    break
        else:
            new_temp_var["name"] += "_{suffix}1".format(suffix=suffix)

        # create a new variable
        new_var = hint.from_dict(new_temp_var)
        return new_var

    def _get_temp_var(self, hint: VarBase) -> VarBase:
        return self._get_var_by(hint, "tmp")

    # This function seems unncessary
    # def _get_var_decollision(self, hint: VarBase) -> VarBase:
    #     return self._get_var_by(hint, "decollision")

    def get_temp_var(self, hint: VarBase) -> VarBase:
        """
        This method creates and returns a new variable, and add it to the table.
        It can be used during the pattern matching process that lowers Inter Op
        DSL to Inter Op SSA.
        At that time only variables names are registered in the variable table,
        and all the rest of the information in the variable table are not
        produced yet.
        """
        # TODO: get shape
        new_var = self._get_temp_var(hint)
        # self.numbered_val_to_name[new_var.to_string()] = new_var.to_string()
        # self.numbered_name_vals[new_var.to_string()] = [new_var.to_string()]
        self.register_var(new_var)
        return new_var

    def register_var(self, var: VarBase):
        """
        This method registers a variable name. This is done to every op result,
        i.e., def op result, during the lowering from Inter Op DSL to Inter Op
        SSA in order to have knowledge about the existing variable names. This
        is necessary to make sure during creating temporary variable names, no
        name collision happens.
        """
        self.numbered_val_to_name[var.to_string()] = var.to_string()
        self.numbered_name_vals[var.to_string()] = [var.to_string()]

    def get_numbered_val(self, hint: VarBase) -> VarBase:
        """
        This method creates and returns a new variable indicating numbered value
        of hint, and add it to the table.
        This method is to be used after the Inter Op SSA program is complete,
        i.e., after the lowering from Inter Op DSL to Inter Op SSA.
        At this time all the operations from the statements should be ready for
        analysis. And this process is usually called during the value numbering,
        when def-use chain analysis has been done.
        """
        new_var = self._get_temp_var(hint)
        self.numbered_val_to_name[new_var.to_string()] = hint.to_string()
        self.numbered_name_vals[hint.to_string()].append(new_var.to_string())
        return new_var


def calc_op_to_seq(operations: list[Union[OpBase, FusedOpBase]]) -> dict[OpBase, int]:
    """calculate the operation to sequence id mapping. Fused op will be broken
    down into basic ops and each will be assigned a unique id"""
    op_to_seq: dict[OpBase, int] = dict()
    curr_idx = 0
    for op in operations:
        if isinstance(op, FusedOpBase):
            for sub_op in op.ops:
                op_to_seq[sub_op] = curr_idx
                curr_idx += 1
        else:
            op_to_seq[op] = curr_idx
        curr_idx += 1
    return op_to_seq


class Program:
    operations: list[Union[OpBase, FusedOpBase]]
    op_to_seq: dict[OpBase, int]
    var_table: VariableTable

    def __init__(
        self, var_table: VariableTable, operations: list[Union[OpBase, FusedOpBase]]
    ):
        self.var_table = var_table
        self.operations = operations
        self.op_to_seq = calc_op_to_seq(operations)

    # TODO: get use-def chain
    def get_users_of_result(self, operation: OpBase) -> list[OpBase]:
        raise NotImplementedError

    def get_defining_op(self, var: str) -> OpBase:
        raise NotImplementedError

    def get_seqid(self, op: OpBase) -> int:
        """returns the sequence id of the operation"""
        assert op in self.operations
        return self.operations.index(op)

    def assert_define_before_use(self, operand: VarBase, op: OpBase):
        assert operand in self.var_table.numbered_val_to_name
        # operand should either be a weight, or defined before
        if not isinstance(operand, WeightVar):
            assert self.get_seqid(self.get_defining_op(operand)) < self.get_seqid(op)

    def validate(self) -> None:
        # returns True if 1) every operation has all key-value pairs correctly
        # defined as specified in this file, and 2) use-def chain is correct
        for var in self.var_table.numbered_val_to_name:
            assert self.get_defining_op(var) is not None
            assert is_valid_var_name(var)
        for op in self.operations:
            op.validate()
            if isinstance(op, FusedOpBase):
                for op_ in op.ops:
                    for operand in op_.get_operands():
                        self.assert_define_before_use(operand, op_)
            else:
                for operand in op.get_operands():
                    self.assert_define_before_use(operand, op)
        return

    def dumps(self) -> str:
        """Returns a multi-line string that contains the variable shape table,
        and operations. In other words, the result is in the format of
        .inter-op-ssa file"""
        result = ""
        result += self.var_table.dumps()
        result += "\nDAG{"
        for op in self.operations:
            result += op.to_string()
            result += "\n"
        result += "}\n"
        return result

    @classmethod
    def loads(cls, lines: list[str]) -> "Program":
        var_table, ops = program_serializer.program_loads(lines)

        return cls(var_table, ops)

    def differentiate(self) -> "Program":
        """
        differentiate the program, and return the differentiated program
        """
        diff_var_table = VariableTable()
        diff_ops: list[Union[OpBase, FusedOpBase]] = []
        for op in self.operations:
            diff_ops += op.differentiate()
        # Reconstruct the variable table
        # Notice that if the differentiation is done after forward pass value
        # numbering, the value number chain of the same name may not be preserved
        for op in diff_ops:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    for result in sub_op.get_results():
                        diff_var_table.numbered_val_to_name.add(result.to_string())
            else:
                for result in op.get_results():
                    diff_var_table.numbered_val_to_name.add(result.to_string())
        return Program(diff_var_table, diff_ops)

    def infer_shapes(self):
        """
        after differentiation or pattern match from the inter-op IR, we get all
        operations and unique variable names. We need to infer the shapes of all
        variables.
        """
        raise NotImplementedError
