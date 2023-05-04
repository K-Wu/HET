#!/usr/bin/env python3
import torch
from torch.autograd.function import FunctionCtx
from dgl.backend.pytorch import *
from ..dummy_ctx import *

# sparse.SEGMENTMM


def rgnn_relational_matmul_no_scatter_gather_list(
    ntype_offset_ptrs,
    weights,  # (num node types, num_heads, in_dim, feat_dim)
    inputs,  # (node num, num_heads, in_dim)
    ret,  # (node num, num_heads, feat_dim)
):
    # do the segment mm for each head. Transposing head to the first dimension to make it easier to do the segment mm.
    inputs = torch.transpose(inputs, 0, 1)
    weights = torch.transpose(weights, 0, 1)
    ret_ = torch.zeros(
        inputs.shape[1], inputs.shape[0], weights.shape[2], device=inputs.device
    )
    for i in range(weights.shape[0]):
        ctx = DummyCtx()
        ret_[i, :, :] = sparse.SEGMENTMM.forward(
            ctx, inputs[i], weights[i], ntype_offset_ptrs
        )
    ret[:] = torch.transpose(ret_, 0, 1)


def backward_rgnn_relational_matmul_no_scatter_gather_list(
    ntype_offset_ptrs,
    weights_transposed,
    inputs,  # (node num, num_heads, in_dim)
    gradout,  # (node num, num_heads, feat_dim)
    grad_weight,  # (num_node_types, num_heads, in_dim, feat_dim)
    grad_input,  # (node num, num_heads, in_dim)
):
    grad_input_ = torch.zeros(inputs.shape, device=inputs.device)
    grad_input_ = torch.transpose(grad_input_, 0, 1)
    grad_weight_ = torch.zeros(weights_transposed.shape, device=inputs.device)
    grad_weight_ = torch.transpose(grad_weight_, 0, 1)

    inputs = torch.transpose(inputs, 0, 1)
    weights_transposed = torch.transpose(weights_transposed, 0, 1)
    for i in range(grad_weight_.shape[0]):
        ctx = DummyCtx()
        ctx.save_for_backward(
            inputs[i], torch.transpose(weights_transposed[i], 2, 3), ntype_offset_ptrs
        )
        curr_grad_input_, curr_grad_weight_, _ = sparse.SEGMENTMM.backward(ctx, gradout)
        grad_input_[i, :, :] = curr_grad_input_
        grad_weight_[i, :, :, :] = curr_grad_weight_

    grad_input_ = torch.transpose(grad_input_, 0, 1)
    grad_weight_ = torch.transpose(grad_weight_, 0, 1)
    grad_input[:] = grad_input_
    grad_weight[:] = grad_weight_


def rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    weights,  # (num_edge_types, num_heads, in_dim, feat_dim)
    inputs,  # (node num, num_heads, in_dim)
    ret,  # (num edges, num_heads, feat_dim)
    input_num_head_one_flag,
):
    ret_ = torch.zeros(
        (separate_coo_eids.shape[0], inputs.shape[1], weights.shape[2]),
        device=inputs.device,
    )
    inputs = torch.index_select(inputs, 0, separate_coo_node_indices)
    rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_relptrs, weights, inputs, ret_
    )
    # reorder according to the edge ids
    ret[separate_coo_eids] = ret_


def backward_rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    weights_transposed,  # (num_edge_types, num_heads, feat_dim, in_dim)
    inputs,  # (node num, num_heads, in_dim)
    gradout,  # (num edges, num_heads, feat_dim)
    grad_input,  # (node num, num_heads, in_dim)
    grad_weight,  # (num_heads, in_dim, feat_dim)
    input_num_head_one_flag,
):
    grad_input_per_edge = torch.zeros(
        (gradout.shape[0], inputs.shape[1], inputs.shape[2]), device=inputs.device
    )
    # reorder gradout according to the edge ids
    gradout = gradout[separate_coo_eids]
    backward_rgnn_relational_matmul_no_scatter_gather_list(
        separate_coo_relptrs,
        weights_transposed,
        inputs,
        gradout,
        grad_weight,
        grad_input_per_edge,
    )
    grad_input[:] = torch.index_add(
        grad_input_per_edge, 0, separate_coo_node_indices, grad_input_per_edge
    )


# NB: KWU: reflect the deprecation
# def rgnn_relational_matmul_ac_gather_scatter_list_identical(
#     separate_coo_relptrs,
#     separate_coo_eids,
#     weights,  # (num_edge_types, num_heads, in_dim, feat_dim)
#     inputs,  # (num edges, num_heads, in_dim)
#     ret,  # (num edges, num_heads, feat_dim)
#     input_num_head_one_flag,
# ):
#     rgnn_relational_matmul(
#         separate_coo_relptrs,
#         separate_coo_eids,
#         separate_coo_eids,
#         weights,
#         inputs,
#         ret,
#         input_num_head_one_flag,
#     )


# def backward_rgnn_relational_matmul_ac_gather_scatter_list_identical(
#     separate_coo_relptrs,
#     separate_coo_eids,
#     weights_transposed,  # (num_edge_types, num_heads, feat_dim, in_dim)
#     inputs,  # (num edges, num_heads, in_dim)
#     gradout,  # (num edges, num_heads, feat_dim)
#     grad_input,  # (num edges, num_heads, in_dim)
#     grad_weight,  # (num_heads, in_dim, feat_dim)
#     input_num_head_one_flag,
# ):
#     backward_rgnn_relational_matmul(
#         separate_coo_relptrs,
#         separate_coo_eids,
#         separate_coo_eids,
#         weights_transposed,
#         inputs,
#         gradout,
#         grad_input,
#         grad_weight,
#         input_num_head_one_flag,
#     )


def rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    weight,  # (num_edge_types, num_heads, in_dim, feat_dim)
    node_feat,  # (node num, num_heads, in_dim)
    ret,  # (unique (node, rel) num, num_heads, feat_dim)
    input_num_head_one_flag,
):
    node_feat = torch.index_select(node_feat, 0, unique_srcs_and_dests_node_idx)
    rgnn_relational_matmul_no_scatter_gather_list(
        unique_srcs_and_dests_rel_ptr, weight, node_feat, ret
    )


def backward_rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    weight_transposed,
    node_feat,
    gradout,
    grad_node_feat,
    grad_weight,
    input_num_head_one_flag,
):
    grad_node_feat_ = torch.zeros(
        (
            unique_srcs_and_dests_node_idx.shape[0],
            node_feat.shape[1],
            node_feat.shape[2],
        ),
        device=node_feat.device,
    )
    node_feat = torch.index_select(node_feat, 0, unique_srcs_and_dests_node_idx)
    backward_rgnn_relational_matmul_no_scatter_gather_list(
        unique_srcs_and_dests_rel_ptr,
        weight_transposed,
        node_feat,
        gradout,
        grad_weight,
        grad_node_feat_,
    )
    grad_node_feat[:] = torch.index_add(
        grad_node_feat_, 0, unique_srcs_and_dests_node_idx, grad_node_feat_
    )


def rgnn_inner_product_edge_and_node(
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_edge_data,  # (edge_num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    ret,  # (edge_num, num_heads)
):
    left_edge_data = left_edge_data[separate_coo_eids]
    right_edge_data = torch.index_select(
        right_node_vectors, 0, separate_coo_col_indices, out=right_node_vectors
    )
    feat_dim = left_edge_data.shape[-1]
    num_heads = left_edge_data.shape[-2]
    ret[separate_coo_eids] = torch.reshape(
        torch.bmm(
            torch.reshape(left_edge_data, (-1, 1, feat_dim)),
            torch.reshape(right_edge_data, (-1, feat_dim, 1)),
        ),
        (-1, num_heads),
    )


def backward_rgnn_inner_product_edge_and_node(
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_edge_data,  # (edge_num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    gradout,  # (edge_num, num_heads)
    grad_left_edge_data,
    grad_right_node_vectors,
):
    left_edge_data = left_edge_data[separate_coo_eids]
    right_edge_data = torch.index_select(
        right_node_vectors, 0, separate_coo_col_indices, out=right_node_vectors
    )
    feat_dim = left_edge_data.shape[-1]
    num_heads = left_edge_data.shape[-2]
    grad_left_edge_data[separate_coo_eids] = torch.reshape(
        torch.bmm(
            torch.reshape(gradout, (-1, 1, num_heads)),
            torch.reshape(right_edge_data, (-1, num_heads, feat_dim)),
        ),
        (-1, 1, feat_dim),
    )
    grad_right_node_vectors[:] = torch.index_add(
        grad_right_node_vectors,
        0,
        separate_coo_col_indices,
        torch.reshape(
            torch.bmm(
                torch.reshape(left_edge_data, (-1, 1, feat_dim)),
                torch.reshape(gradout, (-1, num_heads, 1)),
            ),
            (-1, 1, num_heads),
        ),
    )


def towrap_rgnn_inner_product_node_compact_and_node(
    unique_node_idx_inverse_mapping,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_node_compact_data,  # (unique (node, rel) num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    ret,  # (edge_num, num_heads)
):
    # left_node_uncompact = torch.zeros((unique_srcs_and_dests_node_idx.shape[0], left_node_compact_data.shape[1], left_node_compact_data.shape[2]), device=left_node_compact_data.device)
    left_node_uncompact = torch.index_select(
        left_node_compact_data, 0, unique_node_idx_inverse_mapping
    )  # (unique_srcs_and_dests_node_idx.shape[0], left_node_compact_data.shape[1], left_node_compact_data.shape[2])
    rgnn_inner_product_edge_and_node(
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_uncompact,  # (node_num, num_heads, feat_dim)
        right_node_vectors,  # (node_num, num_heads, feat_dim)
        ret,  # (edge_num, num_heads
    )


def towrap_backward_rgnn_inner_product_node_compact_and_node(
    unique_node_idx_inverse_mapping,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_idx,
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_node_compact_data,  # (unique (node, rel) num, num_heads, feat_dim)
    right_node_vectors,  # (node_num, num_heads, feat_dim)
    gradout,  # (edge_num, num_heads)
    grad_left_node_compact_data,
    grad_right_node_vectors,
):
    # left_node_uncompact = torch.zeros((unique_srcs_and_dests_node_idx.shape[0], left_node_compact_data.shape[1], left_node_compact_data.shape[2]), device=left_node_compact_data.device)
    left_node_uncompact = torch.index_select(
        left_node_compact_data, 0, unique_node_idx_inverse_mapping
    )  # (unique_srcs_and_dests_node_idx.shape[0], left_node_compact_data.shape[1], left_node_compact_data.shape[2])
    grad_left_node_uncompact = torch.zeros(
        (
            unique_srcs_and_dests_node_idx.shape[0],
            left_node_compact_data.shape[1],
            left_node_compact_data.shape[2],
        ),
        device=left_node_compact_data.device,
    )
    backward_rgnn_inner_product_edge_and_node(
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_uncompact,  # (node_num, num_heads, feat_dim)
        right_node_vectors,  # (node_num, num_heads, feat_dim)
        gradout,  # (edge_num, num_heads
        grad_left_node_uncompact,
        grad_right_node_vectors,
    )
    grad_left_node_compact_data[:] = torch.index_add(
        grad_left_node_compact_data,
        0,
        unique_node_idx_inverse_mapping,
        grad_left_node_uncompact[separate_coo_eids],
    )
