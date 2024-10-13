#!/usr/bin/env python3
from jinja2 import Environment, FileSystemLoader, select_autoescape
from ...utils import is_pwd_het_dev_root
from .op_specs import TraversalOpSpec, GEMMOpSpec
from ..InterOpSSA.variables import DATA_TYPES, WEIGHT_SLICE_TYPES

assert (
    is_pwd_het_dev_root()
), "Please run this script from het_dev root directory."

env = Environment(
    loader=FileSystemLoader(searchpath="./pyctor/"),
    autoescape=select_autoescape(),
)


# TODO: add common arguments used in all kernels: edge_enumerate, matmul, matmul_rgcn_hgt
# torch::Dict<std::string, at::Tensor> graph_tensors_dict,


def get_name(var_spec: list[str]) -> str:
    if var_spec[0] in DATA_TYPES:
        # This is a data var
        return var_spec[1]
    else:
        # This is a weight var
        assert var_spec[1] in WEIGHT_SLICE_TYPES
        return var_spec[0]


def get_cuda_kernel_formal_arguments(
    op_spec: TraversalOpSpec | GEMMOpSpec,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a dictionary of common arguments in the format of (argument_name, argument_type), and a dictionary of specific arguments in the same form"""

    # TODO: add common arguments
    common_arguments_dict = {}
    raise NotImplementedError

    # Add specific arguments
    if isinstance(op_spec, TraversalOpSpec):
        specific_arguments = op_spec.inputs + op_spec.outputs
    else:
        specific_arguments = [op_spec.left, op_spec.right, op_spec.product]
    specific_arguments_dict = {
        get_name(arg): "float *" for arg in specific_arguments
    }
    return common_arguments_dict, specific_arguments_dict


def get_cpp_launcher_formal_arguments(
    op_spec: TraversalOpSpec | GEMMOpSpec,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a dictionary of common arguments in the format of (argument_name, argument_type), and a dictionary of specific arguments in the same form"""

    # TODO: add common arguments
    common_arguments_dict = {}
    raise NotImplementedError

    # Add specific arguments
    if isinstance(op_spec, TraversalOpSpec):
        specific_arguments = op_spec.inputs + op_spec.outputs
    else:
        specific_arguments = [op_spec.left, op_spec.right, op_spec.product]
    specific_arguments_dict = {
        get_name(arg): "at::Tensor&" for arg in specific_arguments
    }
    return common_arguments_dict, specific_arguments_dict


def get_torch_export_python_cpp_function_name_pair(
    op_spec: TraversalOpSpec | GEMMOpSpec,
) -> tuple[str, str]:
    return f"{op_spec.name}_launcher", f"{op_spec.name}_launcher"


def get_torch_export_statements(
    op_specs: list[TraversalOpSpec | GEMMOpSpec],
) -> str:
    template = env.get_template(
        "ir/OpSpecSSA/templates/torch_export.inc.h.jinja"
    )
    func_python_name_to_cpp_name: dict[str, str] = {}
    for op_spec in op_specs:
        (
            func_python_name,
            func_cpp_name,
        ) = get_torch_export_python_cpp_function_name_pair(op_spec)
        func_python_name_to_cpp_name[func_python_name] = func_cpp_name
    return template.render(
        func_python_name_to_cpp_name=func_python_name_to_cpp_name
    ).strip()


def get_python_launcher_invocation(
    op_spec: TraversalOpSpec | GEMMOpSpec,
) -> str:
    arguments = get_cpp_launcher_formal_arguments(op_spec)[1].keys()
    return f"{op_spec.name}_launcher({','.join(arguments)})"


def get_python_autograd_func_formal_arguments(
    op_specs: list[TraversalOpSpec | GEMMOpSpec],
    backward_op_specs: list[TraversalOpSpec | GEMMOpSpec],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return a tuple of four lists of strings: common_forward_arguments, specific_forward_arguments, common_backward_arguments, specific_backward_arguments"""

    # TODO: add common arguments
    common_forward_arguments = []
    common_backward_arguments = []
    raise NotImplementedError

    # Add specific arguments
    specific_forward_arguments = set()
    specific_backward_arguments = set()
    for op_spec in op_specs:
        current_forward_specific_arguments = get_cpp_launcher_formal_arguments(
            op_spec
        )[1].keys()
        specific_forward_arguments.update(current_forward_specific_arguments)
    for op_spec in backward_op_specs:
        current_backward_specific_arguments = (
            get_cpp_launcher_formal_arguments(op_spec)[1].keys()
        )
        specific_backward_arguments.update(current_backward_specific_arguments)

    return (
        common_forward_arguments,
        sorted(specific_forward_arguments),
        common_backward_arguments,
        sorted(specific_backward_arguments),
    )


def generate_python_forward_stmt_and_autograd_func(
    op_specs: list[TraversalOpSpec | GEMMOpSpec],
    backward_op_specs: list[TraversalOpSpec | GEMMOpSpec],
) -> tuple[str, str]:
    raise NotImplementedError


def generate_python_forward_stmts_and_autograd_funcs(
    op_specs: list[TraversalOpSpec | GEMMOpSpec],
    backward_op_specs: list[TraversalOpSpec | GEMMOpSpec],
    auto_diff_op_pairs: list[tuple[list[int], list[int]]],
) -> tuple[str, str]:
    """Generate the forward statements and autograd functions for a list of op_specs and backward_op_specs."""
    forward_stmts: str = ""
    autograd_funcs: str = ""

    for fwd_op_idxes, bwd_op_idxes in auto_diff_op_pairs:
        fwd_op_specs = list(map(op_specs.__getitem__, fwd_op_idxes))
        bwd_op_specs = list(map(backward_op_specs.__getitem__, bwd_op_idxes))
        (
            current_forward_stmt,
            current_autograd_func,
        ) = generate_python_forward_stmt_and_autograd_func(
            fwd_op_specs, bwd_op_specs
        )
        forward_stmts += current_forward_stmt
        autograd_funcs += current_autograd_func
        forward_stmts += "\n"
        autograd_funcs += "\n"

    return forward_stmts, autograd_funcs


def is_traversal_fall_back(op_spec: TraversalOpSpec) -> bool:
    """If op_spec contains one TraversalSimpleOp and the inputs and output are in the same parallel domain, e.g., in the edge parallel domain or node parallel domain, we may fall back to use PyTorch functions to define the forward and backward functions."""

    raise NotImplementedError


def generate_traversal_cpp_launcher(op_spec: TraversalOpSpec) -> str:
    """This returns one multi-line string that contains both the cpp host function and the torch export statement."""
    template = env.get_template(
        "ir/OpSpecSSA/templates/edge_enumerate.launcher.jinja"
    )
    result_str = ""

    cpp_launcher_name = f"{op_spec.name}_launcher"
    raise NotImplementedError


def generate_traversal_cuda_kernel(op_spec: TraversalOpSpec) -> str:
    template = env.get_template(
        "ir/OpSpecSSA/templates/edge_enumerate.kernel.jinja"
    )
    # return template.render(op_spec=op_spec)
    raise NotImplementedError


def generate_gemm_cpp_launcher(op_spec: GEMMOpSpec) -> str:
    """This returns one multi-line string that contains both the cpp host function and the torch export statement."""
    template = env.get_template("ir/OpSpecSSA/templates/matmul.launcher.jinja")
    template = env.get_template(
        "ir/OpSpecSSA/templates/matmul_rgcn_hgt.launcher.jinja"
    )
    result_str = ""

    cpp_launcher_name = f"{op_spec.name}_launcher"
    raise NotImplementedError


def generate_gemm_cuda_kernel(op_spec: GEMMOpSpec) -> str:
    template = env.get_template("ir/OpSpecSSA/templates/matmul.kernel.jinja")
    template = env.get_template(
        "ir/OpSpecSSA/templates/matmul_rgcn_hgt.kernel.jinja"
    )
    # return template.render(op_spec=op_spec)
    raise NotImplementedError


def codgen_main_procedure():
    raise NotImplementedError
