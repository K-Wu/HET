#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class RgnnRelationalMatmulACScatterGatherListIdentical(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_relptrs,
        separate_coo_eids,
        weights,
        inputs,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
        )
        K.rgnn_relational_matmul_ac_gather_scatter_list_identical(
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
            ret,
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
        grad_weight = th.zeros_like(weights)
        grad_input = th.zeros_like(inputs)
        K.rgnn_relational_matmul_backward_ac_gather_scatter_list_identical(
            separate_coo_relptrs,
            separate_coo_eids,
            th.transpose(weights, 2, 3),
            inputs,
            gradout,
            grad_input,
            grad_weight,
        )
        # fmt: off
        return None, None, grad_weight, grad_input, None
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
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            weights,
            inputs,
        )
        K.rgnn_relational_matmul(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            weights,
            inputs,
            ret,
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
        grad_weight = th.zeros_like(weights)
        grad_input = th.zeros_like(inputs)
        # FIXME: seems there is a deadlock here
        K.rgnn_relational_matmul_backward(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            th.transpose(weights, 2, 3),
            inputs,
            gradout,
            grad_input,
            grad_weight,
        )
        # fmt: off
        return None, None, None, grad_weight, grad_input, None
        # fmt: on


def rgnn_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indices,
    separate_coo_eids,
    # unique_srcs_and_dests_rel_ptr,
    # unique_srcs_and_dests_node_indices,
    weights,
    inputs,
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
    )
    if separate_coo_node_indices.data_ptr() == separate_coo_eids.data_ptr():
        return RgnnRelationalMatmulACScatterGatherListIdentical.apply(
            separate_coo_relptrs,
            separate_coo_eids,
            weights,
            inputs,
            ret,
        )
    else:
        return RgnnRelationalMatmul.apply(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            # unique_srcs_and_dests_rel_ptr,
            # unique_srcs_and_dests_node_indices,
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
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
        )
        K.rgnn_relational_matmul_compact_as_of_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
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
        grad_weight = th.zeros_like(weight)
        grad_node_feat = th.zeros_like(node_feat)
        K.backward_rgnn_relational_matmul_compact_as_of_node(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            th.transpose(weight, 2, 3),
            node_feat,
            ret,
            gradout,
            grad_weight,
            grad_node_feat,
        )
        # fmt: off
        return None, None, grad_weight, grad_node_feat, None
        # fmt: on


def rgnn_relational_matmul_compact_as_of_node(
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    weight,
    node_feat,
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
    )
