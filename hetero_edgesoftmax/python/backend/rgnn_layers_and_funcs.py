#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


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
            {
                "separate_coo_rel_ptrs": separate_coo_relptrs,
                "separate_coo_node_indices": separate_coo_node_indices,
                "separate_coo_eids": separate_coo_eids,
            },
            0,  # CompactAsOfNodeKind::Disabled
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
        grad_weight = th.zeros_like(
            weights, memory_format=th.contiguous_format)
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)
        # FIXME: seems there is a deadlock here
        K.backward_rgnn_relational_matmul(
            {
                "separate_coo_rel_ptrs": separate_coo_relptrs,
                "separate_coo_node_indices": separate_coo_node_indices,
                "separate_coo_eids": separate_coo_eids,
            },
            0,  # CompactAsOfNodeKind:Disabled
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
        grad_weight = th.zeros_like(
            weights, memory_format=th.contiguous_format)
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)
        # FIXME: seems there is a deadlock here
        K.backward_rgnn_relational_matmul_no_scatter_gather_list(
            ntype_offset_ptrs,
            th.transpose(weights, 2, 3).contiguous(),
            inputs,
            gradout.contiguous(),
            grad_input,
            grad_weight,
        )
        # print("grad_weight", grad_weight)
        # fmt: off
        return None,  grad_weight, grad_input, None
        # fmt: on


class RgnnRelationalMatmulCompactAsOfNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptrs,
        unique_srcs_and_dests_node_indices,
        weight,
        node_feat,
        ret,
        input_num_head_one_flag,
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptrs,
            unique_srcs_and_dests_node_indices,
            weight,
            node_feat,
            ret,
        )
        ctx.input_num_head_one_flag = input_num_head_one_flag

        K.rgnn_relational_matmul(
            {
                "unique_srcs_and_dests_rel_ptrs": unique_srcs_and_dests_rel_ptrs,
                "unique_srcs_and_dests_node_indices": unique_srcs_and_dests_node_indices,
            },
            1,  # CompactAsOfNodeKind:Enabled
            weight,
            node_feat,
            ret,
            input_num_head_one_flag,
        )
        # print(unique_srcs_and_dests_rel_ptrs)
        # print(unique_srcs_and_dests_node_indices)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            unique_srcs_and_dests_rel_ptrs,
            unique_srcs_and_dests_node_indices,
            weight,
            node_feat,
            ret,
        ) = ctx.saved_tensors
        input_num_head_one_flag = ctx.input_num_head_one_flag
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        grad_node_feat = th.zeros_like(
            node_feat, memory_format=th.contiguous_format)
        # FIXME: illegal reading A data when BGS compactAsOfNode is true perhaps bug in schedule_by_relation
        # FIXME: seems there is a bug in gradout when bgs compactAsOfNode is true perhaps the scheming scheme to grad_feat_src is faulty
        # print(unique_srcs_and_dests_rel_ptrs)
        # print(unique_srcs_and_dests_node_indices)
        K.backward_rgnn_relational_matmul(
            {
                "unique_srcs_and_dests_rel_ptrs": unique_srcs_and_dests_rel_ptrs,
                "unique_srcs_and_dests_node_indices": unique_srcs_and_dests_node_indices,
            },
            1,  # CompactAsOfNodeKind::Enabled
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


class RgnnInnerProductNodeCompactAndNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptrs,
        unique_srcs_and_dests_node_indices,
        separate_coo_rel_ptrs,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_compact_data,
        right_node_vectors,
        ret,
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptrs,
            unique_srcs_and_dests_node_indices,
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
        K.rgnn_inner_product_right_node_separatecoo(
            {
                "unique_srcs_and_dests_rel_ptrs": unique_srcs_and_dests_rel_ptrs,
                "unique_srcs_and_dests_node_indices": unique_srcs_and_dests_node_indices,
            },
            1,  # CompactAsOfNodeKind::Enabled
            separate_coo_rel_ptrs,
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
            unique_srcs_and_dests_rel_ptrs,
            unique_srcs_and_dests_node_indices,
            separate_coo_rel_ptrs,
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
        K.backward_inner_product_right_node_separatecoo(
            {
                "unique_srcs_and_dests_rel_ptrs": unique_srcs_and_dests_rel_ptrs,
                "unique_srcs_and_dests_node_indices": unique_srcs_and_dests_node_indices,
            },
            1,  # CompactAsOfNodeKind::Enabled
            separate_coo_rel_ptrs,
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


class RgnnInnerProductNodeCompactAndNodeWithDirectIndexing(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        edata_index_to_inverse_index,
        separate_coo_rel_ptrs,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_node_compact_data,
        right_node_vectors,
        ret,
    ):
        ctx.save_for_backward(
            edata_index_to_inverse_index,
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
        K.rgnn_inner_product_right_node_separatecoo(
            {
                "edata_idx_to_inverse_idx": edata_index_to_inverse_index,
            },
            2,  # CompactAsOfNodeKind::EnabledWithDirectIndexing
            separate_coo_rel_ptrs,
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
            edata_index_to_inverse_index,
            separate_coo_rel_ptrs,
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
        K.backward_inner_product_right_node_separatecoo(
            {
                "edata_idx_to_inverse_idx": edata_index_to_inverse_index,
            },
            2,  # CompactAsOfNodeKind::EnabledWithDirectIndexing
            separate_coo_rel_ptrs,
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
        return None, None, None, None,  None, grad_left_node_compact_data, grad_right_node_vectors, None
        # fmt: on


class RgnnInnerProductEdgeAndNode(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_rel_ptrs,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        left_edge_data,
        right_node_vectors,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            ret,
        )
        K.rgnn_inner_product_right_node_separatecoo(
            {},
            0,  # CompactAsOfNodeKind::Disabled
            separate_coo_rel_ptrs,
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
            separate_coo_rel_ptrs,
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
        K.backward_inner_product_right_node_separatecoo(
            dict(),
            0,  # CompactAsOfNodeKind::Disabled
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            left_edge_data,
            right_node_vectors,
            gradout.contiguous(),
            grad_left_edge_data,
            grad_right_node_vectors,
        )
        # FIXME: check if there is bug here
        # fmt: off
        return None, None, None,  None, grad_left_edge_data, grad_right_node_vectors, None
        # fmt: on


# TODO: add direct indexing support
# TODO: merge this with rgnn_relational_matmul_compact_as_of_node
def rgnn_relational_matmul(
    arg_tensor_dict, weights, inputs, input_num_head_one_flag, compact_as_of_node_kind
):
    if compact_as_of_node_kind == 1:  # CompactAsOfNodeKind::Enabled
        ret = th.zeros(
            (
                int(arg_tensor_dict["unique_srcs_and_dests_rel_ptrs"][-1]),
                weights.size(1),
                weights.size(3),
            ),
            dtype=weights.dtype,
            device=weights.device,
            requires_grad=True,
        )
        return RgnnRelationalMatmulCompactAsOfNode.apply(
            arg_tensor_dict["unique_srcs_and_dests_rel_ptrs"],
            arg_tensor_dict["unique_srcs_and_dests_node_indices"],
            weights,
            inputs,  # originally node_feat
            ret,
            input_num_head_one_flag,
        )

    elif compact_as_of_node_kind == 0:  # CompactAsOfNodeKind::Disabled
        ret = th.zeros(
            (
                arg_tensor_dict["separate_coo_node_indices"].numel(),
                weights.size(1),
                weights.size(3),
            ),  # [num_items, num_heads, out_feats//num_heads]
            dtype=weights.dtype,
            device=weights.device,
            requires_grad=True,
        ).contiguous()
        return RgnnRelationalMatmul.apply(
            arg_tensor_dict["separate_coo_rel_ptrs"],
            arg_tensor_dict["separate_coo_node_indices"],
            arg_tensor_dict["separate_coo_eids"],
            weights.contiguous(),
            inputs.contiguous(),
            ret,
            input_num_head_one_flag,
        )
    else:
        raise NotImplementedError


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
    )  # .contiguous()
    return RgnnRelationalMatmulNoScatterGatherList.apply(
        ntype_offset_ptrs,
        weights,
        inputs,
        ret,
    )


# TODO: this is now using dst as left-hand side data and src as right-hand side data


def rgnn_inner_product_right_node(
    graph,
    left_side_data,
    right_node_vectors,
    compact_as_of_node_kind,
    left_mapper_suffix: str,  # _row or _col or None to determine tensor keys
):
    if compact_as_of_node_kind == 0:  # CompactAsOfNodeKind::Disabled
        left_edge_data = left_side_data
        separate_coo_original_dict = graph.get_separate_coo_original()
        # assuming shape of right_node_vectors is [num_nodes, num_heads, num_features]
        ret = th.zeros(
            [separate_coo_original_dict["eids"].numel(), right_node_vectors.size(1)],
            dtype=right_node_vectors.dtype,
            device=right_node_vectors.device,
            requires_grad=True,
        ).contiguous()
        return RgnnInnerProductEdgeAndNode.apply(
            separate_coo_original_dict["rel_ptrs"],
            separate_coo_original_dict["eids"],
            separate_coo_original_dict["row_indices"],
            separate_coo_original_dict["col_indices"],
            left_edge_data,
            right_node_vectors,
            ret,
        )
    elif compact_as_of_node_kind == 1:  # CompactAsOfNodeKind::Enabled
        left_node_compact_data = left_side_data
        separate_coo_original_dict = graph.get_separate_coo_original()
        separate_unique_node_indices_dict_single_sided = (
            graph.get_separate_unique_node_indices_single_sided()
        )
        # assuming shape of right_node_vectors is [num_nodes, num_heads, num_features]
        # print([separate_coo_original_dict["rel_ptrs"][-1], right_node_vectors.size(1)])
        ret = th.zeros(
            (separate_coo_original_dict["rel_ptrs"]
             [-1], right_node_vectors.size(1)),
            dtype=right_node_vectors.dtype,
            device=right_node_vectors.device,
            requires_grad=True,
        ).contiguous()
        return RgnnInnerProductNodeCompactAndNode.apply(
            separate_unique_node_indices_dict_single_sided[
                "rel_ptrs" + left_mapper_suffix
            ],
            separate_unique_node_indices_dict_single_sided[
                "node_indices" + left_mapper_suffix
            ],
            separate_coo_original_dict["rel_ptrs"],
            separate_coo_original_dict["eids"],
            separate_coo_original_dict["row_indices"],
            separate_coo_original_dict["col_indices"],
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
    elif compact_as_of_node_kind == 2:  # CompactAsOfNodeKind::EnabledWIthDirectIndexing
        left_node_compact_data = left_side_data
        separate_unique_node_indices_dict_single_sided_inverse_idx = (
            graph.get_separate_unique_node_indices_single_sided_inverse_idx()
        )
        separate_coo_original_dict = graph.get_separate_coo_original()
        # assuming shape of right_node_vectors is [num_nodes, num_heads, num_features]
        ret = th.zeros(
            [separate_coo_original_dict["rel_ptrs"]
                [-1], right_node_vectors.size(1)],
            dtype=right_node_vectors.dtype,
            device=right_node_vectors.device,
            requires_grad=True,
        ).contiguous()
        return RgnnInnerProductNodeCompactAndNodeWithDirectIndexing.apply(
            separate_unique_node_indices_dict_single_sided_inverse_idx[
                "inverse_indices" + left_mapper_suffix
            ],
            separate_coo_original_dict["rel_ptrs"],
            separate_coo_original_dict["eids"],
            separate_coo_original_dict["row_indices"],
            separate_coo_original_dict["col_indices"],
            left_node_compact_data,
            right_node_vectors,
            ret,
        )
    else:
        raise NotImplementedError
