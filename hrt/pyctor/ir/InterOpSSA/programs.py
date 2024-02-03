#!/usr/bin/env python3
from .variables import (
    VarBase,
    is_valid_var_name,
    DataVar,
    WeightVar,
    parse_var_class,
    Shape,
)
from .operators import OpBase, FusedOpBase, UnrealizedBinaryOp
from typing import (
    Union,
    NamedTuple,
    Annotated,
    Generator,
    Callable,
    Generic,
    TypeVar,
)
import traceback
import re
from functools import wraps
from recordclass import dataobject


class CallRecord(dataobject):
    callstack: list[str]
    funcname: str
    msg: str


# From hrt/misc/playground/try_print_call_site.py and https://stackoverflow.com/questions/60219591/using-a-paramaterized-decorator-for-recording-methods-in-a-class
def log_pass_calls(description: str):
    """Decorate class functions that do analysis or transform pass and record the call site in the called_function list of the class instance"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = ("STARTED | FUNCTION: {} | ARGS: {} | KWARGS: {} ").format(
                func.__name__, args, kwargs
            )
            # print(msg)

            args[0].passes_call_records.append(
                CallRecord(
                    callstack=traceback.format_stack(),
                    funcname=func.__name__,
                    msg=msg,
                )
            )  # i.e., self.called_function

            return func(*args, **kwargs)

        return wrapper

    return decorator


T = TypeVar("T")


class MySet(set[T], Generic[T]):
    """
    Set that records analysis passes and transform passes.
    Example:
    ```
    class Program:
        analysis_passes: MySet[Callable]
        transform_passes: MySet[Callable]

        @transform_passes.register
        def do_something(self):
            ...

        @analysis_passes.register
        def check_something(self):
            ...
    ```
    From https://stackoverflow.com/questions/50372342/class-with-a-registry-of-methods-based-on-decorators
    """

    def register(self, method):
        self.add(method)
        return method


def strip_white_spaces(line: str) -> str:
    """Strip whitespaces from line"""
    return re.sub(r"\s+", "", line)


DefUseEntry = NamedTuple(
    "DefUseEntry",
    [
        ("name", str),
        ("def_op", Union[OpBase, None]),
        ("use_ops", list[OpBase]),
    ],
)


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

    passes_call_records: list[
        CallRecord
    ]  # Stores the logging by @log_pass_calls
    passes: MySet[Callable] = MySet()  # Stores analysis and transform passes

    vars_input: set[VarBase]

    vars_shape: dict[VarBase, Shape]
    dsl_vars: Annotated[
        set[VarBase],
        "variables defined during the lowering from Inter Op DSL to SSA",
    ]
    numbered_val_to_key: Annotated[
        dict[VarBase, VarBase],
        """map the full string representation to full string representation, e.g., (EDGEWISE, "var_name2") to (EDGEWISE, "var_name")""",
    ]
    numbered_key_vals: Annotated[
        dict[VarBase, list[VarBase]],
        """
    reverse of numbered_val_to_key""",
    ]
    def_use_table: Annotated[
        # after SSA numbering, each value will be a single DefUseEntry
        dict[VarBase, Union[DefUseEntry, list[DefUseEntry]]],
        """
    numbered_key_vals and def_use_table together store the value numbering information

    numbered_key_vals
    before value numbering, each (key, value) in numbered_key_vals looks like
    (var_key, [var_name]) (another_var, [another_var]) ...
    after value numbering, the above entry may now look like
    (var_key, [var_name, var_name2, var_name3]) (another_var, [another_var])

    def_use_table
    before value numbering, each (key, value) in def_use_table looks like
    (var_key, [DefUseEntry(var_name, opid0, [opid1]), DefUseEntry(var_name, opid2, [opid3, opid4])])
    (another_var, [DefUseEntry(another_var, opid5, [opid6])])
    after value numbering, the above entry may now look like
    (var_name, DefUseEntry(var_name, opid0, opid1)),
    (var_name2, DefUseEntry(var_name2, opid2, [opid3, opid4]))
    (another_var, DefUseEntry(another_var, opid5, [opid6]))
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

    @classmethod
    def loads(cls, lines: list[str]) -> "VariableTable":
        """
        initiate the variable table by loading the shape info in the text
        :param lines begin with "ShapeTable{", and end with "}". To read a file,
        specify this parameter as fd.readlines()
        For now, we assume nothing else left in the first and the last line.
        Adapted from loads_op from hrt/pyctor/ir/InterOpSSA/serialize_program.py
        """
        var_table = cls()

        assert strip_white_spaces(lines[0].strip()) == "ShapeTable{"
        assert strip_white_spaces(lines[-1].strip()) == "}"
        lines = lines[1:-1]

        # load initial variables and weights

        from . import program_serializer

        scopes = program_serializer.find_first_level_scopes(lines)
        for scope_beg, scope_end, scope_tag in scopes:
            if scope_tag == "InitialVariablesAndWeights":
                for line in lines[scope_beg + 1 : scope_end]:
                    line_var_strs = line.replace(")", "))").split(")")
                    line_var_strs = [
                        var_str.strip() for var_str in line_var_strs
                    ]
                    line_var_strs = [
                        var_str[1:]
                        for var_str in line_var_strs
                        if var_str[0] == ","
                    ]
                    line_var_strs = {
                        var_str.strip() for var_str in line_var_strs
                    }
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
                    var_table.vars_shape[var] = Shape(type=shape)

                    var_table.numbered_key_vals[var] = [var]
                    var_table.numbered_val_to_key[var] = var
                    for varname in varnames:
                        var_newer = var.get_numbered_var(varname)
                        var_table.numbered_key_vals[var].append(var_newer)
                        var_table.numbered_val_to_key[var_newer] = var
                    var_table.vars_shape[var] = Shape(type=shape)
        return cls(var_table)

    def dumps(self) -> str:
        """output the variable table in the text, i.e., the shape table"""
        result = "ShapeTable{\n"
        # Step 1: Output initial variables and weights
        result += "InitialVariablesAndWeights{\n"
        result += ", ".join([var.to_string() for var in self.vars_input])
        result += "\n}\n"

        # Step 2: Output shape info
        result += "VariableNumbersAndShapes{\n"
        for var, shape in self.vars_shape.items():
            result += f"{shape.type}: {var.to_string()}"
            if var in self.numbered_key_vals:
                for var_newer in self.numbered_key_vals[var][1:]:
                    result += f" <- {var_newer.to_string()}"
            result += "\n"
        result += "}\n"

        result += "}"

        return result

    # TODO: implement shape in this table
    def get_shape_info(self, var: VarBase) -> Union[Shape, None]:
        """get the shape of a variable"""
        key = self.get_var_key(var)
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
        key = self.get_var_key(var)
        if key in self.vars_shape and self.vars_shape[key] != new_shape_info:
            raise ValueError(
                f"Variable {var.get_name()} already has a shape"
                f" {self.vars_shape[key]}, cannot set to {new_shape_info}"
            )
        self.vars_shape[key] = new_shape_info

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

        # create a new variable
        new_var = hint.__class__.from_dict(new_temp_var)
        return new_var

    def _get_temp_var(self, hint: VarBase) -> VarBase:
        return self._get_var_by(hint, "tmp")

    # This function seems unncessary
    # def _get_var_decollision(self, hint: VarBase) -> VarBase:
    #     return self._get_var_by(hint, "decollision")

    def get_temp_var_dsl(self, hint: VarBase) -> VarBase:
        """
        This method creates and returns a new variable, and add it to the table.
        It can be used during the pattern matching process that lowers Inter Op
        DSL to Inter Op SSA.
        At that time only variables names are registered in the variable table,
        and all the rest of the information in the variable table are not
        produced yet.
        """
        new_var = self._get_temp_var(hint)
        self.register_dsl_var(new_var)
        return new_var

    def register_input_and_weight_var(self, var: VarBase) -> None:
        """
        This method is called to register a variable that is the input data or weight variable
        """
        self.vars_input.add(var)
        self.register_value_zero(var)

    def register_dsl_var(self, var: VarBase) -> None:
        """
        This method registers a variable key. This is done to every op result,
        i.e., def op result, during the lowering from Inter Op DSL to Inter Op
        SSA in order to have knowledge about the existing variable names. This
        is necessary to make sure during creating temporary variable names, no
        name collision happens.
        """
        # self.numbered_val_to_name[var.to_string()] = var.to_string()
        # self.numbered_name_vals[var.to_string()] = [var.to_string()]
        self.dsl_vars.add(var)

    def register_value_zero(self, var: VarBase) -> None:
        """
        Register the first value of a variable name.
        """
        self.numbered_key_vals[var] = [var]
        self.numbered_val_to_key[var] = var

    def increase_and_register_value_number(self, var: VarBase) -> VarBase:
        """
        This method creates and returns a new variable indicating numbered value
        of var, and add it to the table.
        This method is to be used after the Inter Op SSA program is complete,
        i.e., after the lowering from Inter Op DSL to Inter Op SSA.
        At this time all the operations from the statements should be ready for
        analysis. And this process is usually called during the value numbering,
        when def-use chain analysis has been done.
        """
        # TODO: for now, we reuse the _tmp suffix to number values
        new_var = self._get_temp_var(var)
        self.numbered_val_to_key[new_var] = var
        self.numbered_key_vals[var].append(new_var)
        return new_var

    @classmethod
    def _do_value_number_on_program(
        cls, ops: list[OpBase]
    ) -> tuple["VariableTable", list[OpBase]]:
        """
        This method does value numbering on all the operations in a program.
        numbered_key_vals and numbered_val_to_key will be updated accordingly.
        Notice that this function should only be applied on unnumbered program,
        otherwise it will malfunction.
        """
        var_table = cls()
        new_ops = []
        for op in ops:
            new_op: OpBase = op
            # Use set to deduplicate for cases where one operand/result shows multiple times, so that only one replacement for all these occurrence will be applied
            for opr in {*op.get_operands()}:
                # For operands, use the latest numbered value
                if opr in var_table.numbered_key_vals:
                    new_opr = var_table.numbered_key_vals[opr][-1]
                    new_op = new_op.replace_all_operands_with(opr, new_opr)
                else:
                    # register the variable as data input or weights
                    var_table.register_input_and_weight_var(opr)

            for opr in {*op.get_results()}:
                # For results, increase the value number if already defined
                if opr in var_table.numbered_key_vals:
                    # increment the number
                    new_opr = var_table.increase_and_register_value_number(opr)
                    new_op = new_op.replace_all_results_with(opr, new_opr)
                else:
                    # register the 0th value of the variable
                    var_table.register_value_zero(opr)
            new_ops.append(new_op)
        return var_table, new_ops

    @passes.register
    @log_pass_calls("do_value_number_on_program")
    def do_value_number_on_program(self, ops: list[OpBase]) -> list[OpBase]:
        new_var_table, new_ops = self._do_value_number_on_program(ops)
        self.numbered_key_vals = new_var_table.numbered_key_vals
        self.numbered_val_to_key = new_var_table.numbered_val_to_key
        self.vars_input = new_var_table.vars_input
        return new_ops

    @passes.register
    @log_pass_calls("do_data_input_and_weight_var_analysis")
    def do_data_input_and_weight_var_analysis(self, ops: list[OpBase]) -> None:
        new_var_table, _ = self._do_value_number_on_program(ops)
        self.vars_input = new_var_table.vars_input
        return

    @passes.register
    @log_pass_calls("do_def_use_chain_analysis")
    def do_def_use_chain_analysis(
        self, ops: list[OpBase], after_value_numbering: bool
    ) -> None:
        """
        This method does def-use chain analysis on all the operations in a
        program, and creates def_use_table.
        No fused op is allowed.
        """
        if len(self.def_use_table) != 0:
            print("Warning: def_use_table is not empty, will be overwritten.")
        self.def_use_table = dict()

        # Step 1 create dummy entry for input variables and weight variables
        if not after_value_numbering:
            # TODO: skip this step if this pass is already done though not after_value_numbering
            self.do_data_input_and_weight_var_analysis(ops)
        for var in self.vars_input:
            # Set def_op as none to indicate input and weight variables
            entry = DefUseEntry(name=var.get_name(), def_op=None, use_ops=[])
            if after_value_numbering:
                self.def_use_table[var] = entry
            else:
                self.def_use_table[var] = [entry]

        # Step 2 process every operation
        for op in ops:
            # Each definition corresponds to one DefUseEntry.
            # Before ssa numbering is done, key value pair in the dict is (var_key, list[DefUseEntry])
            # After ssa numbering is done, key value pair in the dict is (value (var_namen), DefUseEntry)
            for res in {*op.get_results()}:
                entry = DefUseEntry(name=res.get_name(), def_op=op, use_ops=[])
                # Whether after_value_numbering is True or not, we don't need to (calculate and ) refer to numbered_val_to_key to find the dictionary key
                if after_value_numbering:
                    self.def_use_table[res] = entry
                else:
                    if res not in self.def_use_table:
                        self.def_use_table[res] = []
                    dict_record = self.def_use_table[res]
                    assert isinstance(dict_record, list)
                    dict_record.append(entry)

            for opr in {*op.get_operands()}:
                # Whether after_value_numbering is True or not, we can use opr.to_string() as the dictionary key
                assert opr in self.def_use_table
                dict_record = self.def_use_table[opr]
                if after_value_numbering:
                    assert isinstance(dict_record, DefUseEntry)
                    dict_record.use_ops.append(op)
                else:
                    assert isinstance(dict_record, list)
                    dict_record[-1].use_ops.append(op)

    def differentiate(
        self, diff_ops: list[Union[OpBase, FusedOpBase]]
    ) -> "VariableTable":
        raise NotImplementedError


def calc_op_to_seq(
    operations: list[Union[OpBase, FusedOpBase]]
) -> dict[OpBase, int]:
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
    passes_call_records: list[
        CallRecord
    ]  # Stores the logging by @log_pass_calls
    passes: MySet[Callable] = MySet()  # Stores analysis and transform passes

    operations: list[Union[OpBase, FusedOpBase]]
    op_to_seq: dict[
        OpBase, int
    ]  # fused op is broken down into basic ops in this dict
    var_table: VariableTable

    def __init__(
        self,
        var_table: VariableTable,
        operations: list[Union[OpBase, FusedOpBase]],
    ):
        self.var_table = var_table
        self.operations = operations
        self.op_to_seq = calc_op_to_seq(operations)

    # TODO: remove if not used
    # def get_users_of_result(self, operation: OpBase) -> list[OpBase]:
    #     raise NotImplementedError

    def get_defining_op(self, var: VarBase) -> Union[OpBase, None]:
        """returns the operation that defines the variable.
        This function will return None if the variable is an input or weight variable,
        and this function will raise Error if the variable is not found in the program.
        """
        if var not in self.var_table.numbered_val_to_key:
            raise ValueError(
                f"Variable {var} is not found in this program. Make sure the"
                " analysis is run before calling get_defining_op!"
            )
        if isinstance(self.var_table.def_use_table[var], list):
            for entry in self.var_table.def_use_table[var]:
                assert isinstance(entry, DefUseEntry)
                if entry.name == var.get_name():
                    return entry.def_op
        else:
            entry = self.var_table.def_use_table[var]
            assert isinstance(entry, DefUseEntry)
            return entry.def_op

    def get_seqid(self, op: OpBase) -> int:
        """returns the sequence id of the operation"""
        assert op in self.operations
        return self.operations.index(op)

    def assert_define_before_use(self, operand: VarBase, op: OpBase):
        assert operand in self.var_table.numbered_val_to_key
        # operand should either be a weight, or defined before
        if not isinstance(operand, WeightVar):
            operand_def_op = self.get_defining_op(operand)
            if operand_def_op is None:
                # this is an input variable or weight. Skipping the seqid check
                return

            assert self.get_seqid(operand_def_op) < self.get_seqid(op)

    def validate(self) -> None:
        # returns True if 1) every operation has all key-value pairs correctly
        # defined as specified in this file, and 2) use-def chain is correct
        for var in self.var_table.numbered_val_to_key:
            assert self.get_defining_op(var) is not None
            assert is_valid_var_name(var.get_name())
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
        from . import program_serializer

        scopes: list[
            tuple[int, int, str]
        ] = program_serializer.find_first_level_scopes(lines)
        var_table = VariableTable.loads(lines[scopes[0][0] : scopes[0][1] + 1])
        ops = program_serializer.loads_op(lines[scopes[1][0] :])

        return cls(var_table, ops)

    @passes.register
    @log_pass_calls("differentiate")
    def differentiate(self) -> "Program":
        """
        differentiate the program, and return the differentiated program
        """
        diff_ops: list[Union[OpBase, FusedOpBase]] = []
        for op in self.operations:
            diff_ops += op.differentiate()
        # Reconstruct the variable table
        # Notice that if the differentiation is done after forward pass value
        # numbering, the value number chain of the same key may not be preserved
        diff_var_table = self.var_table.differentiate(diff_ops)
        return Program(diff_var_table, diff_ops)

    def yield_basic_ops(self) -> Generator[OpBase, None, None]:
        for op in self.operations:
            if isinstance(op, FusedOpBase):
                for sub_op in op.ops:
                    yield sub_op
            else:
                yield op

    @passes.register
    @log_pass_calls("realize_ops")
    def realize_ops(self):
        """Realize Unrealized.* ops after shape has been inferred"""

        op_replacement: dict[OpBase, OpBase] = dict()
        for op in self.yield_basic_ops():
            if isinstance(op, UnrealizedBinaryOp):
                curr_opr_shape_info = [
                    self.var_table.get_shape_info(opr)
                    for opr in op.get_operands()
                ]
                curr_res_shape_info = [
                    self.var_table.get_shape_info(res)
                    for res in op.get_results()
                ]
                new_op = op.realize(curr_opr_shape_info, curr_res_shape_info)
                op_replacement[op] = new_op

        # Replace the op with the new in self.operations
        for idx, op in enumerate(self.operations):
            if isinstance(op, FusedOpBase):
                for idx_sub, sub_op in enumerate(op.ops):
                    if op in op_replacement:
                        op.ops[idx_sub] = op_replacement[op]
            else:
                if op in op_replacement:
                    self.operations[idx] = op_replacement[op]

        # Redo the shape analysis if it is done before
        done_infer_shape_flag = False
        for pass_record in self.var_table.passes_call_records:
            if self.infer_shapes.__name__ == pass_record.funcname:
                done_infer_shape_flag = True
        if done_infer_shape_flag:
            self.infer_shapes()

        # Redo the def-use chain analysis if it is done before
        done_value_numbering_flag = False
        done_def_use_chain_analysis = False
        for pass_record in self.var_table.passes_call_records:
            if (
                self.var_table.do_def_use_chain_analysis.__name__
                == pass_record.funcname
            ):
                done_def_use_chain_analysis = True
            if (
                self.var_table.do_data_input_and_weight_var_analysis.__name__
                == (pass_record.funcname)
            ):
                done_value_numbering_flag = True
        if done_def_use_chain_analysis:
            self.var_table.do_def_use_chain_analysis(
                list(self.yield_basic_ops()), done_value_numbering_flag
            )

    @passes.register
    @log_pass_calls("infer_shapes")
    def infer_shapes(self):
        """
        after differentiation or pattern match from the inter-op IR, we get all
        operations and unique variable names. We need to infer the shapes of all
        variables.
        """
        for op in self.yield_basic_ops():
            curr_opr_shape_info = [
                self.var_table.get_shape_info(opr) for opr in op.get_operands()
            ]
            curr_res_shape_info = [
                self.var_table.get_shape_info(res) for res in op.get_results()
            ]

            opr_shape_info, res_shape_info = op.infer_shape(
                curr_opr_shape_info, curr_res_shape_info
            )
            for opr, shape_info in zip(op.get_operands(), opr_shape_info):
                if shape_info is not None:
                    self.var_table.set_shape_info_or_throw(opr, shape_info)
            for res, shape_info in zip(op.get_results(), res_shape_info):
                if shape_info is not None:
                    self.var_table.set_shape_info_or_throw(res, shape_info)

        raise NotImplementedError
