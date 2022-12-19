#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class RgnnRelationalMatmulACScatterGatherListIdentical(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_relptrs,
        separate_coo_eids,
        weights,
        inputs,
        ret,
        input_num_head_one_flag,
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
        )
        ctx.input_num_head_one_flag = input_num_head_one_flag
        K.rgnn_relational_matmul_ac_gather_scatter_list_identical(
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
            ret,
            input_num_head_one_flag,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
        ) = ctx.saved_tensors
        input_num_head_one_flag = ctx.input_num_head_one_flag
        grad_weight = th.zeros_like(weights, memory_format=th.contiguous_format)
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)
        K.backward_rgnn_relational_matmul_ac_gather_scatter_list_identical(
            separate_coo_relptrs,
            separate_coo_eids,
            th.transpose(weights, 2, 3).contiguous(),
            inputs,
            gradout.contiguous(),
            grad_input,
            grad_weight,
            input_num_head_one_flag,
        )
        # fmt: off
        return None, None, grad_weight, grad_input, None, None
        # fmt: on


class RgnnRelationalMatmul(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_relptrs,
        separate_coo_node_indices,
        separate_coo_eids,
        weights,
        inputs,
        ret,
        input_num_head_one_flag,
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            weights,
            inputs,
        )
        ctx.input_num_head_one_flag = input_num_head_one_flag
        K.rgnn_relational_matmul(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            weights,
            inputs,
            ret,
            input_num_head_one_flag,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            weights,
            inputs,
        ) = ctx.saved_tensors
        input_num_head_one_flag = ctx.input_num_head_one_flag
        grad_weight = th.zeros_like(weights, memory_format=th.contiguous_format)
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)
        # FIXME: seems there is a deadlock here
        K.backward_rgnn_relational_matmul(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            th.transpose(weights, 2, 3).contiguous(),
            inputs,
            gradout.contiguous(),
            grad_input,
            grad_weight,
            input_num_head_one_flag,
        )
        # fmt: off
        return None, None, None, grad_weight, grad_input, None, None
        # fmt: on


def rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    # unique_srcs_and_dests_rel_ptr,
    # unique_srcs_and_dests_node_indices,
    weights,
    inputs,
    input_num_head_one_flag,
):
    ret = th.zeros(
        (
            separate_coo_node_indices.numel(),
            weights.size(1),
            weights.size(3),
        ),  # [num_items, num_heads, out_feats//num_heads]
        dtype=weights.dtype,
        device=weights.device,
        requires_grad=True,
    ).contiguous()
    if separate_coo_node_indices.data_ptr() == separate_coo_eids.data_ptr():
        return RgnnRelationalMatmulACScatterGatherListIdentical.apply(
            separate_coo_relptrs,
            separate_coo_eids,
            weights.contiguous(),
            inputs.contiguous(),
            ret,
            input_num_head_one_flag,
        )
    else:
        return RgnnRelationalMatmul.apply(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            # unique_srcs_and_dests_rel_ptr,
            # unique_srcs_and_dests_node_indices,
            weights.contiguous(),
            inputs.contiguous(),
            ret,
            input_num_head_one_flag,
        )


class RgnnRelationalMatmulNoScatterGatherList(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ntype_offset_ptrs,
        weights,
        inputs,
        ret,
    ):

        K.rgnn_relational_matmul_no_scatter_gather_list(
            ntype_offset_ptrs,
            weights,
            inputs,
            ret,
        )
        ctx.save_for_backward(ntype_offset_ptrs, weights, inputs, ret)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            ntype_offset_ptrs,
            weights,
            inputs,
            node_feat_output,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weights, memory_format=th.contiguous_format)
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)
        # FIXME: seems there is a deadlock here
        K.backward_rgnn_relational_matmul_no_scatter_gather_list(
            ntype_offset_ptrs,
            th.transpose(weights, 2, 3).contiguous(),
            node_feat_output,
            gradout.contiguous(),
            grad_weight,
            grad_input,
        )
        # fmt: off
        return None,  grad_weight, grad_input, None
        # fmt: on


def rgnn_relational_matmul_no_scatter_gather_list(
    ntype_offset_ptrs,
    weights,
    inputs,
):
    ret = th.zeros(
        (
            inputs.size(0),
            weights.size(3),
        ),  # [num_items, out_feats]
        dtype=weights.dtype,
        device=weights.device,
        # requires_grad=True,
    )
    return RgnnRelationalMatmulNoScatterGatherList.apply(
        ntype_offset_ptrs,
        weights,
        inputs,
        ret,
    )


class RgnnRelationalMatmulCompactAsOfNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_idx,
        weight,
        node_feat,
        ret,
        input_num_head_one_flag,
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
        )
        ctx.input_num_head_one_flag = input_num_head_one_flag
        K.rgnn_relational_matmul_compact_as_of_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
            input_num_head_one_flag,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
        ) = ctx.saved_tensors
        input_num_head_one_flag = ctx.input_num_head_one_flag
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        grad_node_feat = th.zeros_like(node_feat, memory_format=th.contiguous_format)
        # FIXME: illegal reading A data when BGS compactAsOfNode is true perhaps bug in schedule_by_relation
        # FIXME: seems there is a bug in gradout when bgs compactAsOfNode is true perhaps the scheming scheme to grad_feat_src is faulty
        K.backward_rgnn_relational_matmul_compact_as_of_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            th.transpose(weight, 2, 3).contiguous(),
            node_feat,
            gradout.contiguous(),
            grad_node_feat,
            grad_weight,
            input_num_head_one_flag,
        )
        # fmt: off
        return None, None, grad_weight, grad_node_feat, None, None
        # fmt: on


def rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    weight,
    node_feat,
    input_num_head_one_flag,
):
    ret = th.zeros(
        [unique_srcs_and_dests_rel_ptr[-1], weight.size(1), weight.size(3)],
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgnnRelationalMatmulCompactAsOfNode.apply(
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        weight,
        node_feat,
        ret,
        input_num_head_one_flag,
    )


class RgnnRelationalMatmulCompactAsOfNodeSingleEnded(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_idx,
        separate_coo_rel_ptr,
        separate_coo_node_indices,
        separate_coo_eids,
        weight,
        node_feat,
        ret,
        input_num_head_one_flag,
    ):
        node_feat = node_feat.contiguous()
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_node_indices,
            separate_coo_eids,
            weight,
            node_feat,
            ret,
        )
        ctx.input_num_head_one_flag = input_num_head_one_flag
        K.rgnn_relational_matmul_compact_as_of_node_single_ended(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_node_indices,
            separate_coo_eids,
            weight,
            node_feat,
            ret,
            input_num_head_one_flag,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_node_indices,
            separate_coo_eids,
            weight,
            node_feat,
            ret,
        ) = ctx.saved_tensors
        input_num_head_one_flag = ctx.input_num_head_one_flag
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        grad_node_feat = th.zeros_like(node_feat, memory_format=th.contiguous_format)
        K.backward_rgnn_relational_matmul_compact_as_of_node_single_ended(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_node_indices,
            separate_coo_eids,
            th.transpose(weight, 2, 3).contiguous(),
            node_feat,
            ret,
            gradout.contiguous(),
            grad_weight,
            grad_node_feat,
            input_num_head_one_flag,
        )
        # fmt: off
        return None, None, None, None, None,  grad_weight, grad_node_feat, None, None
        # fmt: on


def rgnn_relational_matmul_compact_as_of_node_single_ended(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    separate_coo_rel_ptr,
    separate_coo_node_indices,
    separate_coo_eids,
    weight,
    node_feat,
    input_num_head_one_flag,
):
    ret = th.empty(
        [separate_coo_rel_ptr[-1], weight.size(1), weight.size(3)],
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
        memory_format=th.contiguous_format,
    )
    return RgnnRelationalMatmulCompactAsOfNodeSingleEnded.apply(
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        separate_coo_rel_ptr,
        separate_coo_node_indices,
        separate_coo_eids,
        weight,
        node_feat,
        ret,
        input_num_head_one_flag,
    )


class RgnnInnerProductNodeCompactAndNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_idx,
        separate_coo_rel_ptr,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_compact_data,
        right_node_vectors,
        ret,
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
        K.rgnn_inner_product_node_compact_and_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            ret,
        ) = ctx.saved_tensors
        grad_left_node_compact_data = th.zeros_like(
            left_node_compact_data, memory_format=th.contiguous_format
        )
        grad_right_node_vectors = th.zeros_like(
            right_node_vectors, memory_format=th.contiguous_format
        )
        K.backward_rgnn_inner_product_node_compact_and_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            gradout.contiguous(),
            grad_left_node_compact_data,
            grad_right_node_vectors,
        )
        # fmt: off
        return None, None, None, None, None,  None, grad_left_node_compact_data, grad_right_node_vectors, None
        # fmt: on


def rgnn_inner_product_node_compact_and_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_node_compact_data,
    right_node_vectors,
):
    # assuming shape of right_node_vectors is [num_nodes, num_heads, num_features]
    ret = th.zeros(
        [separate_coo_rel_ptr[-1], right_node_vectors.size(1)],
        dtype=right_node_vectors.dtype,
        device=right_node_vectors.device,
        requires_grad=True,
    )
    return RgnnInnerProductNodeCompactAndNode.apply(
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        separate_coo_rel_ptr,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_compact_data,
        right_node_vectors,
        ret,
    )


class RgnnInnerProductEdgeAndNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_edge_data,
        right_node_vectors,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            ret,
        )
        K.rgnn_inner_product_edge_and_node(
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            ret,
        ) = ctx.saved_tensors
        grad_left_edge_data = th.zeros_like(
            left_edge_data, memory_format=th.contiguous_format
        )
        grad_right_node_vectors = th.zeros_like(
            right_node_vectors, memory_format=th.contiguous_format
        )
        K.backward_rgnn_inner_product_edge_and_node(
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            gradout.contiguous(),
            grad_left_edge_data,
            grad_right_node_vectors,
        )
        print(grad_left_edge_data)
        print(grad_right_node_vectors)
        # fmt: off
        return None, None, None, grad_left_edge_data, grad_right_node_vectors, None
        # fmt: on


def rgnn_inner_product_edge_and_node(
    separate_coo_eids,
    separate_coo_row_indices,
    separate_coo_col_indices,
    left_edge_data,
    right_node_vectors,
):
    # assuming shape of right_node_vectors is [num_nodes, num_heads, num_features]
    ret = th.zeros(
        [separate_coo_eids.numel(), right_node_vectors.size(1)],
        dtype=right_node_vectors.dtype,
        device=right_node_vectors.device,
        requires_grad=True,
    )
    return RgnnInnerProductEdgeAndNode.apply(
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_edge_data,
        right_node_vectors,
        ret,
    )
