from .environ import env
from ...InterOpSSA.variable_tables import VariableTable
from ...InterOpSSA.variables import parse_var_spec_class, VarBase, Shape
from ..op_specs import (
    TraversalLoopOpSpec,
    TraversalSimpleOpSpec,
    TraversalOpSpec,
)
from typing import Optional
import enum
from ....utils.logger import logger

## The following strings are used in edge_enumerate_kernel.cu.h.codelet.jinja
OP_TO_CPP_STATEMENT = {
    "mul": "{{output}}={{inputs0}}*{{inputs1}}",
    "negative": "{{output}}=-{{inputs0}}",
    "divide": "{{output}}={{inputs0}}/{{inputs1}}",
    "sum_accumulation": "{{output}}+={{inputs0}}",
    "sum_accumulation_warp": """
    DType accumulation_warp_tmp = 0.;
    
    accumulation_warp_tmp += {{inputs0}};
    
    /// PYCTOR: Warp reduction
    for (Idx sum_idx = hidden_xlen; sum_idx > 0; sum_idx >>= 1) {
        accumulation_warp_tmp += __shfl_down_sync(0xffffffff, accumulation_warp_tmp, sum_idx);
    }
    if (threadIdx.x % hidden_xlen == 0) {
        {{output}} = accumulation_warp_tmp;
    }
    """,
}


CPP_CUDA_KERNEL_PROLOGUE = """
Idx resume_from = 0;
// Idx num_heads = gdata.num_heads;
{% if is_type2_schedule %}
Idx hidden_xlen = feat_src_xlen / num_heads;
{% endif %}
"""

CPP_EDGE_ENUMERATION_LOOP_BODY_PROLOGUE = """
Idx dst_vid = row_indices[eidx];
Idx edata_idx = eids[eidx];
Idx src_vid = column_indices[eidx];
"""

CPP_FOR_SCOPE_BEGIN = (
    "for({{loop_variable}}={{loop_begin}};{{loop_variable}}<{{loop_end}};{{loop_variable}}+={{loop_step}}) {"
)
CPP_SCOPE_END = "}"

CPP_SCALAR_TEMPLATE_DECLARATION = "DType {{var_name}} = 0.0;"
CPP_EDGEWISE_VECTOR_INDEX_SCHEME = """
constexpr bool EtypeRelPtrIndexSearch = true;
{% if kind.is_compact() %}
    // TODO: etype is not needed if etype_mapper_data's kind is subject
    // to !IsBinarySearch(kind)
    if constexpr (ETypeRelPtrFlag) {
        if constexpr (EtypeRelPtrIndexSearch) {
            etype = linear_search(etype_data.num_relations, etype_data.etypes,
                                eidx, resume_from);
            resume_from = etype;
        } else {
            etype = binary_search(etype_data.num_relations, etype_data.etypes,
                                eidx);
        }
    } else {
        etype = etype_data.etypes[eidx];
    }
{% endif %}
Idx feat_src_entry_id = -1;
{% if kind.is_compact() %}
    // TODO: etype is not needed if etype_mapper_data's kind is subject
    // to !IsBinarySearch(kind)
    feat_src_entry_id = find_relational_compact_as_of_node_index(
        etype, src_vid, edata_idx, etype_mapper_data);
{% else %}
    // NB: we need to use edata_idx instead of eidx here
    feat_src_entry_id = edata_idx;
{% endif %}
"""

_CPP_SIMPLE_LOOP_TEMPLATE = """
/// PYCTOR: Loop header and prologue
{{ for_scopes_begin_and_body_prologue }}
// An example of body prologue:
// Idx dst_vid = row_indices[eidx];
// Idx edata_idx = gdata.eids[eidx];
// Idx src_vid = column_indices[eidx];
{{ loop_scalar_temp_variables_definition }}
{{ indexing_scheme }}
{{ operation_and_warp_reduction_statement}}
{{ for_scopes_end }}
"""


# Enum class equivalent to CompactAsOfNodeKind in hrt/include/kernel_enums.h
class CompactAsOfNodeKind(enum.Enum):
    Disabled = "Disabled"
    Enabled = "Enabled"
    EnabledWithDirectIndexing = "EnabledWithDirectIndexing"
    EnabledWithDualList = "EnabledWithDualList"
    EnabledWithDualListWithDirectIndexing = (
        "EnabledWithDualListWithDirectIndexing"
    )

    def __eq__(self, other):
        if isinstance(other, CompactAsOfNodeKind):
            return self.value == other.value
        elif isinstance(other, str):
            return self.value == other
        return False

    def is_compact(self):
        return self != CompactAsOfNodeKind.Disabled

    def is_binary_search(self):
        return (
            self == CompactAsOfNodeKind.Enabled
            or self == CompactAsOfNodeKind.EnabledWithDirectIndexing
        )

    def print(self):
        return self.name


def _any_compact_shape(vartable: VariableTable) -> bool:
    for var_key_str in vartable.vars_shape:
        if vartable.vars_shape[var_key_str].row_purpose == "unique_node_etype":
            return True
    return False


def get_kind(vartable: VariableTable) -> CompactAsOfNodeKind:
    if _any_compact_shape(vartable):
        # TODO: Check only the variables involved in this operator instead of all variables in the program
        kind = CompactAsOfNodeKind.EnabledWithDualListWithDirectIndexing
    else:
        kind = CompactAsOfNodeKind.Disabled
    return kind


def _generate_kernel_loop_op(
    op_spec: TraversalLoopOpSpec, var_table: VariableTable
) -> str:
    result = env.from_string(CPP_FOR_SCOPE_BEGIN).render(
        loop_variable=op_spec.loop_variable,
        loop_begin=op_spec.loop_begin,
        loop_end=op_spec.loop_end,
        loop_step=op_spec.loop_step,
    )
    result += "\n"

    # Initialize temporary scalar variables
    if len(op_spec.loop_scalar_tmps) > 0:
        for scalar_tmp in op_spec.loop_scalar_tmps:
            result += env.from_string(CPP_SCALAR_TEMPLATE_DECLARATION).render(
                var_name=scalar_tmp
            )
            result += "\n"

    # Generate indexing scheme if the current loop level is edgewise
    assert op_spec.loop_variable in [
        "idx_edge",
        "idx_node",
        "idx_head",
        "idx_feature",
    ]
    if op_spec.loop_variable == "idx_edge":
        # Get kind
        kind = get_kind(var_table)
        # Generate indexing scheme
        result += env.from_string(CPP_EDGEWISE_VECTOR_INDEX_SCHEME).render(
            kind=kind
        )
        result += "\n"

    for op in op_spec.operators:
        if isinstance(op, TraversalLoopOpSpec):
            result += _generate_kernel_loop_op(op, var_table)
        else:
            # It is a TraversalSimpleOpSpec
            result += _generate_kernel_simple_op(op)

    result += CPP_SCOPE_END
    result += "\n"
    return result


def get_indexed_var(
    var: list[str],
    vartable: VariableTable,
    dimension_removed_by_broadcast: Optional[str] = None,
) -> str:
    variable: VarBase = parse_var_spec_class(var).from_opspec_list(var)
    if variable.name.startswith("loop_scalar_tmp"):
        shape = Shape(row_purpose=variable.type, slice_type="scalar")
        # No need to index
        return variable.name
    else:
        shape = vartable.get_shape_info(variable)

    # Get the list denoting each dimension from the leading dimension to the innermost dimension
    dimensions_max = []
    dimensions_idx = []

    if shape.row_purpose in {"nodewise", "srcnode"}:
        # The index of source node index is src_vid
        # TODO: for backward propagation, we need to use in_csr instead of out_csr
        dimensions_idx.append("src_vid")
    elif shape.row_purpose == "dstnode":
        # The index of destination node index is dst_vid
        dimensions_idx.append("dst_vid")
    elif shape.row_purpose == "edgewise":
        # The index of edata is edata_idx
        dimensions_idx.append("edata_idx")
    elif shape.row_purpose == "unique_node_etype":
        dimensions_idx.append("feat_src_entry_id")
    elif shape.row_purpose == "edgetype":
        # The index of edge type is etype
        dimensions_idx.append("etype")
    else:
        raise NotImplementedError(
            f"Unsupported row_purpose: {shape.row_purpose}"
        )

    # TODO: support num_heads>1
    if shape.slice_type == "scalar":
        dimensions_idx += ["0"]
        dimensions_max += ["num_heads"]
    elif shape.slice_type == "vector":
        dimensions_idx += ["0", "idx_feature"]
        dimensions_max += ["num_heads", "num_features"]
    elif shape.slice_type == "matrix":
        dimensions_idx += ["0", "idx_feature", "idx_head"]
        dimensions_max += ["num_heads", "num_features", "num_heads"]

    # Handle broadcast
    # If broadcast added a new dimension, e.g., edgewise, dstnode, nothing additional need to be done
    # Otherwise, if there is a dimension removed by broadcasting, we need to remove the broadcast dimension
    if dimension_removed_by_broadcast is not None:
        raise NotImplementedError("Not implemented yet")

    dimensions_max_prod = [
        "*".join(list(reversed(dimensions_max))[:idx])
        for idx in range(1, len(dimensions_max) + 1)
    ]
    dimensions_max_prod = ["1"] + dimensions_max_prod
    dimensions_max_prod = list(reversed(dimensions_max_prod))

    return (
        variable.name
        + "["
        + "+".join(
            [
                f"{dim}*{idx}"
                for dim, idx in zip(dimensions_max_prod, dimensions_idx)
            ]
        )
        + "]"
    )


def _generate_kernel_simple_op(
    op_spec: TraversalSimpleOpSpec, var_table: VariableTable
) -> str:
    op_type = op_spec.op
    if (
        isinstance(op_spec.op_type, dict)
        and "type" in op_spec.op_type
        and op_spec.op_type["type"] == "accumulation"
    ):
        op_type += "_accumulation_warp"
    template = env.from_string(OP_TO_CPP_STATEMENT[op_spec.op])
    if len(op_spec.inputs) != 2:
        assert len(op_spec.inputs) == 1
        return template.render(
            output=op_spec.output.get_name(),
            inputs0=get_indexed_var(op_spec.inputs[0], var_table),
        )
    else:
        return template.render(
            output=op_spec.output.get_name(),
            inputs0=get_indexed_var(op_spec.inputs[0], var_table),
            inputs1=get_indexed_var(op_spec.inputs[1], var_table),
        )


def generate_kernel_traversal(
    op_spec: TraversalOpSpec, var_table: VariableTable
) -> str:
    template = env.get_template(
        "ir/OpSpecSSA/templates/edge_parallel_enumerate.kernel.cu.h.codelet.jinja"
    )
    prologue = env.from_string(CPP_CUDA_KERNEL_PROLOGUE).render(
        is_type2_schedule=op_spec.schedule == "type2"
    )
    main_loop = ""
    for op in op_spec.operators:
        if isinstance(op, TraversalLoopOpSpec):
            main_loop += _generate_kernel_loop_op(op, var_table)
        else:
            # It is a TraversalSimpleOpSpec
            main_loop += _generate_kernel_simple_op(op, var_table)

    args_and_types: list[tuple[str, str]] = []
    for var in op_spec.outputs + op_spec.inputs:
        args_and_types.append((var.get_name(), "float *"))

    return template.render(
        kernel_cuda_func_name="traversal_" + str(op_spec.op_idx),
        args_and_types=args_and_types,
        prologue=prologue,
        main_loop=main_loop,
    )


def generate_launcher_traversal(
    op_spec: TraversalOpSpec, var_table: VariableTable
) -> str:
    tensor_args: list[str] = []
    tensor_pointer_args: list[str] = []
    for var in op_spec.outputs + op_spec.inputs:
        # TODO: for backward propagation kernel, order tensor args as output1, grad_output1, output2, grad_output2, ..., input1, grad_input1, ...
        tensor_args.append(var.get_name())
        tensor_pointer_args.append(var.get_name() + ".data_ptr<DType>()")

    variable_definitions = ""
    # num_heads is needed for both type1 schedule and type2 schedule
    found_num_heads = False
    for var in op_spec.outputs + op_spec.inputs:
        variable: VarBase = parse_var_spec_class(var).from_opspec_list(var)
        shape = var_table.get_shape_info(variable)
        if shape.slice_type == "scalar":
            variable_definitions += (
                f"Idx num_heads = SeastarComputeXLength<>({var.get_name()});\n"
            )
            found_num_heads = True
            break
    if not found_num_heads:
        variable_definitions += "Idx num_heads = 1;\n"
        logger.warning(
            "num_heads is set to 1 because it is not found in the variable"
            " table"
        )

    # feat_src_xlen is needed only for type2 schedule
    found_feat_src_xlen = False
    for var in op_spec.outputs + op_spec.inputs:
        variable: VarBase = parse_var_spec_class(var).from_opspec_list(var)
        shape = var_table.get_shape_info(variable)
        if shape.slice_type == "vector":
            variable_definitions += (
                "Idx feat_src_xlen ="
                f" SeastarComputeXLength<>({var.get_name()});\n"
            )
            found_feat_src_xlen = True
            break
    if not found_feat_src_xlen:
        logger.error(
            "feat_src_xlen is not found in the variable table, and it is"
            " not set"
        )

    return env.get_template(
        "ir/OpSpecSSA/templates/edge_parallel_enumerate.launcher.cu.h.codelet.jinja"
    ).render(
        kernel_cuda_func_name="traversal_" + str(op_spec.op_idx),
        tensor_args=tensor_args,
        tensor_pointer_args=tensor_pointer_args,
        # TODO: Support more formats
        IntegratedFormatRatherThanSeparateFlag=False,
        CSRRatherThanCOORFlag=False,
        is_type2_schedule=op_spec.schedule == "type2",
        kind=get_kind(var_table),
        variable_definitions=variable_definitions,
    )
