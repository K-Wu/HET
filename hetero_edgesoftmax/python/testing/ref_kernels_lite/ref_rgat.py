#!/usr/bin/env python3
import torch


def relational_fused_gat_kernel_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num_edges, num_heads)
    er,  # (num_edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num_edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    # exp = exp(leaky_relu(el + er, slope))
    # s+= exp
    # ret = feat_src * exp / s
    exp_ = torch.nn.functional.leaky_relu(el + er, slope)
    exp[:] = torch.exp(exp_)
    exp = exp[separate_coo_eids]
    s[:] = torch.index_add(s, 0, separate_coo_col_idx, exp)
    ret = torch.index_add(
        ret,
        0,
        separate_coo_col_idx,
        feat_src[separate_coo_row_idx]
        * exp.unsqueeze(-1)
        / s[separate_coo_row_idx].unsqueeze(-1),
    )


def backward_relational_fused_gat_separate_coo(
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num_edges, num_heads)
    er,  # (num_edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num_edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num_edges, num_heads)
    grad_er,  # (num_edges, num_heads)
    slope,
):
    # grad_exp = grad_out * (feat_src - ret) / sum
    # grad_el += grad_exp * gradLeaky(el + er, slope)
    # grad_er += grad_exp * gradLeaky(el + er, slope)
    # grad_feat_src += grad_out * exp / sum

    # fixme: check if separate_coo is transposed in bck prop kernels
    # fixme: bck prop needs inverse eids in CUDA kernels
    grad_exp = (
        gradout[separate_coo_col_idx]
        * (feat_src[separate_coo_row_idx] - ret[separate_coo_col_idx])
        / s[separate_coo_col_idx].unsqueeze(-1)
    )
    grad_el[separate_coo_eids] = grad_exp * ((el + er >= 0) * 1 + slope)
    grad_er[separate_coo_eids] = grad_exp * ((el + er >= 0) * 1 + slope)
    grad_feat_src_ = torch.index_add(
        grad_feat_src,
        0,
        separate_coo_row_idx,
        gradout[separate_coo_col_idx]
        * exp.unsqueeze(-1)
        / s[separate_coo_col_idx].unsqueeze(-1),
    )
    grad_feat_src[:] = grad_feat_src_


def towrap_relational_fused_gat_kernel_compact_as_of_node_separate_coo_dual_unique_node_list(
    unique_node_idx_inverse_mapping_row,
    unique_node_idx_inverse_mapping_col,
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr_row,
    separate_unique_node_idx_rel_ptr_col,
    separate_unique_node_idx_node_idx_row,
    separate_unique_node_idx_node_idx_col,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edge, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    el_uncompact = torch.index_select(
        el, 0, unique_node_idx_inverse_mapping_row
    )  # (separate_coo_eids.shape[0], el.shape[1])
    er_uncompact = torch.index_select(
        er, 0, unique_node_idx_inverse_mapping_col
    )  # (separate_coo_eids.shape[0], er.shape[1])
    return relational_fused_gat_kernel_separate_coo(
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_idx,
        separate_coo_col_idx,
        feat_src,  # (num_nodes, num_heads, in_feat)
        el_uncompact,  # (num_edges, num_heads)
        er_uncompact,  # (num_edges, num_heads)
        s,  # (num_nodes, num_heads)
        exp,  # (num_edges, num_heads)
        ret,  # (num_nodes, num_heads, out_feat)
        slope,
    )


def towrap_backward_relational_fused_gat_compact_as_of_node_separate_coo_dual_unique_node_list(
    unique_node_idx_inverse_mapping_row,
    unique_node_idx_inverse_mapping_col,
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr_row,
    separate_unique_node_idx_rel_ptr_col,
    separate_unique_node_idx_node_idx_row,
    separate_unique_node_idx_node_idx_col,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num (etype, unique node idx), num_heads)
    grad_er,  # (num (etype, unique node idx), num_heads)
    slope,
):
    el_uncompact = torch.index_select(
        el, 0, unique_node_idx_inverse_mapping_row
    )  # (separate_coo_eids.shape[0], el.shape[1])
    er_uncompact = torch.index_select(
        er, 0, unique_node_idx_inverse_mapping_col
    )  # (separate_coo_eids.shape[0], er.shape[1])
    grad_el_uncompact = torch.zeros(
        (separate_coo_eids.shape[0], grad_el.shape[1]),
        dtype=grad_el.dtype,
        device=grad_el.device,
    )
    grad_er_uncompact = torch.zeros(
        (separate_coo_eids.shape[0], grad_er.shape[1]),
        dtype=grad_er.dtype,
        device=grad_er.device,
    )
    backward_relational_fused_gat_separate_coo(
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_idx,
        separate_coo_col_idx,
        feat_src,  # (num_nodes, num_heads, in_feat)
        el_uncompact,  # (num_edges, num_heads)
        er_uncompact,  # (num_edges, num_heads)
        s,  # (num_nodes, num_heads)
        exp,  # (num_edges, num_heads)
        ret,  # (num_nodes, num_heads, out_feat)
        gradout,  # (num_nodes, num_heads, out_feat)
        grad_feat_src,  # (num_nodes, num_heads, in_feat)
        grad_el_uncompact,  # (num_edges, num_heads)
        grad_er_uncompact,  # (num_edges, num_heads)
        slope,
    )

    grad_el[:] = torch.index_add(
        grad_el, 0, unique_node_idx_inverse_mapping_row, grad_el_uncompact
    )
    grad_er[:] = torch.index_add(
        grad_er, 0, unique_node_idx_inverse_mapping_col, grad_er_uncompact
    )


def towrap_relational_fused_gat_kernel_compact_as_of_node_separate_coo(
    unique_node_idx_inverse_mapping,
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    el_uncompact = torch.index_select(
        el, 0, unique_node_idx_inverse_mapping
    )  # (separate_coo_eids.shape[0], el.shape[1])
    er_uncompact = torch.index_select(
        er, 0, unique_node_idx_inverse_mapping
    )  # (separate_coo_eids.shape[0], el.shape[1])
    exp_uncompact = torch.index_select(
        exp, 0, unique_node_idx_inverse_mapping
    )  # (separate_coo_eids.shape[0], el.shape[1])
    relational_fused_gat_kernel_separate_coo(
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_idx,
        separate_coo_col_idx,
        feat_src,  # (num_nodes, num_heads, in_feat)
        el_uncompact,  # (num_edges, num_heads)
        er_uncompact,  # (num_edges, num_heads)
        s,  # (num_nodes, num_heads)
        exp_uncompact,  # (num_edges, num_heads)
        ret,  # (num_nodes, num_heads, out_feat)
        slope,
    )


def towrap_backward_relational_fused_gat_compact_as_of_node_separate_coo(
    unique_node_idx_inverse_mapping,
    separate_coo_eids,
    separate_coo_rel_ptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_unique_node_idx_rel_ptr,
    separate_unique_node_idx_node_idx,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num (etype, unique node idx), num_heads)
    er,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num (etype, unique node idx), num_heads)
    grad_er,  # (num (etype, unique node idx), num_heads)
    slope,
):
    grad_el_uncompact = torch.index_select(
        grad_el, 0, unique_node_idx_inverse_mapping
    )  # (separate_coo_eids.shape[0], grad_el.shape[1])
    grad_er_uncompact = torch.index_select(
        grad_er, 0, unique_node_idx_inverse_mapping
    )  # (separate_coo_eids.shape[0], grad_er.shape[1])
    backward_relational_fused_gat_separate_coo(
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_idx,
        separate_coo_col_idx,
        feat_src,  # (num_nodes, num_heads, in_feat)
        el,  # (num_edges, num_heads)
        er,  # (num_edges, num_heads)
        s,  # (num_nodes, num_heads)
        exp,  # (num_edges, num_heads)
        ret,  # (num_nodes, num_heads, out_feat)
        gradout,  # (num_nodes, num_heads, out_feat)
        grad_feat_src,  # (num_nodes, num_heads, in_feat)
        grad_el_uncompact,  # (num_edges, num_heads)
        grad_er_uncompact,  # (num_edges, num_heads)
        slope,
    )

    grad_el[:] = torch.index_add(
        grad_el, 0, unique_node_idx_inverse_mapping, grad_el_uncompact
    )
    grad_er[:] = torch.index_add(
        grad_er, 0, unique_node_idx_inverse_mapping, grad_er_uncompact
    )


def relational_fused_gat_kernel_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num edges, num_heads)
    er,  # (num edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    # kernel not used in evaluation
    raise NotImplementedError


def backward_relational_fused_gat_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    feat_src,  # (num_nodes, num_heads, in_feat)
    el,  # (num edges, num_heads)
    er,  # (num edges, num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num edges, num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_src,  # (num_nodes, num_heads, in_feat)
    grad_el,  # (num edges, num_heads)
    grad_er,  # (num edges, num_heads)
    slope,
):
    # kernel not used in evaluation
    raise NotImplementedError


def rgat_relational_fused_gat_compact_as_of_node_csr(
    incsr_row_ptr,
    incsr_col_idx,
    incsr_eids,
    incsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    el_compact,  # (num (etype, unique node idx), num_heads)
    er_compact,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    slope,
):
    # kernel not finished
    # TODO: need to modify kernel to enable single side
    raise NotImplementedError


def backward_rgat_relational_fused_gat_compact_as_of_node_csr(
    outcsr_row_ptr,
    outcsr_col_idx,
    outcsr_eids,
    outcsr_reltypes,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    el_compact,  # (num (etype, unique node idx), num_heads)
    er_compact,  # (num (etype, unique node idx), num_heads)
    s,  # (num_nodes, num_heads)
    exp,  # (num (etype, unique node idx), num_heads)
    ret,  # (num_nodes, num_heads, out_feat)
    gradout,  # (num_nodes, num_heads, out_feat)
    grad_feat_compact,  # (num (etype, unique node idx), num_heads, in_feat)
    grad_el_compact,  # (num (etype, unique node idx), num_heads)
    grad_er_compact,  # (num (etype, unique node idx), num_heads)
    slope,
):
    # kernel not finished
    # TODO: need to modify kernel to enable single side
    raise NotImplementedError
