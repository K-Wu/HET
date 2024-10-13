from ...InterOpSSA.variable_tables import VariableTable
from ...InterOpSSA.variables import parse_var_spec_class, VarBase, Shape
from ..op_specs import GEMMOpSpec
from .enumerate_templates import get_kind
from .environ import env


def generate_kernel_traversal(
    op_spec: GEMMOpSpec, var_table: VariableTable
) -> str:
    template = env.get_template(
        "ir/OpSpecSSA/templates/matmul.kernel.cu.h.codelet.jinja"
    )
    kind = get_kind(var_table)
    kernel_cuda_func_name = "gemm_" + str(op_spec.op_idx)
    return template.render(
        kind=kind, kernel_cuda_func_name=kernel_cuda_func_name
    )


def generate_launcher_traversal(
    op_spec: GEMMOpSpec, var_table: VariableTable
) -> str:
    template = env.get_template(
        "ir/OpSpecSSA/templates/matmul.launcher.cu.h.codelet.jinja"
    )
    kind = get_kind(var_table)
    kernel_cuda_func_name = "gemm_" + str(op_spec.op_idx)
    left_gather_scheme = op_spec.left[0][7:-1]
    product_scatter_scheme = op_spec.product[0][8:-1]
    return template.render(
        kind=kind,
        kernel_cuda_func_name=kernel_cuda_func_name,
        # TODO: extend the gemm kernel template scheme for when input num_heads>1
        InputNumHeadOneFlag=True,
        ACGatherScatterListIdenticalFlag=left_gather_scheme
        == product_scatter_scheme,
    )
