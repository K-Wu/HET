#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class HGTFullGraphHeteroAttentionOps(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        separate_coo_row_idx,
        separate_coo_col_idx,
        separate_coo_eids,
        separate_coo_relptrs,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        attn_score_weight,
    ):
        unnormalized_attn_score = th.zeros(
            (
                separate_coo_row_idx.numel(),
                attn_score_weight.size(1),
            ),  # weight size (self.num_relations, n_heads, self.d_k, self.d_k)
            dtype=attn_score_weight.dtype,
            device=attn_score_weight.device,
            requires_grad=True,
        )

        # edge_norm = th.zeros_like(unnormalized_attn_score, memory_format=th.contiguous_format)
        attn_score_inner_product = th.zeros(
            separate_coo_eids.size(0),
            applied_klinear_node_features.size(1),
            applied_klinear_node_features.size(2),
            dtype=unnormalized_attn_score.dtype,
            device=unnormalized_attn_score.device,
        ).contiguous()
        K.hgt_full_graph_hetero_attention_ops_coo(
            separate_coo_row_idx,
            separate_coo_col_idx,
            separate_coo_eids,
            separate_coo_relptrs,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_weight,
            attn_score_inner_product,
            unnormalized_attn_score,
        )

        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_idx,
            separate_coo_col_idx,
            separate_coo_eids,
            separate_coo_relptrs,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_inner_product,
            attn_score_weight,
            unnormalized_attn_score,
        )
        return unnormalized_attn_score

    @staticmethod
    def backward(ctx, grad_unnorm_attn_score):
        (
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_idx,
            separate_coo_col_idx,
            separate_coo_eids,
            separate_coo_relptrs,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_inner_product,
            attn_score_weight,
            unnormalized_attn_score,
        ) = ctx.saved_tensors
        # print(attn_score_weight.numel())
        grad_attn_weight = th.zeros_like(
            attn_score_weight, memory_format=th.contiguous_format
        )
        grad_k = th.zeros_like(
            applied_klinear_node_features, memory_format=th.contiguous_format
        )
        grad_q = th.zeros_like(
            applied_qlinear_node_features, memory_format=th.contiguous_format
        )
        K.backward_hgt_full_graph_hetero_attention_ops_coo(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_idx,
            separate_coo_col_idx,
            separate_coo_eids,
            separate_coo_relptrs,
            grad_attn_weight,
            th.transpose(attn_score_weight, 2, 3).contiguous(),
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_inner_product,
            grad_unnorm_attn_score,
            grad_k,
            grad_q,
        )
        # print(grad_attn_weight)
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return  None, None, None, None, None, None, None, None,  grad_k, grad_q, grad_attn_weight
        # fmt: on


# FIXME: use outcsr, incsr instead of transposed and original
# class HGTFullGraphMessageMeanAggregationCSR(th.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         row_ptr,
#         col_idx,
#         eids,
#         reltypes,
#         transposed_row_ptr,
#         transposed_col_idx,
#         transposed_eids,
#         transposed_reltypes,
#         message_per_edge,
#         attn_score,
#         ret,
#     ):

#         ctx.save_for_backward(
#             row_ptr,
#             col_idx,
#             eids,
#             reltypes,
#             transposed_row_ptr,
#             transposed_col_idx,
#             transposed_eids,
#             transposed_reltypes,
#             message_per_edge,
#             attn_score,
#         )
#         K.hgt_full_graph_message_mean_aggregation_csr(
#             incsr_row_ptr,
#             incsr_col_idx,
#             incsr_eids,
#             incsr_reltypes,
#             message_per_edge,
#             normalized_attn_score,
#             ret,
#             unique_srcs_and_dests_rel_ptr,
#             unique_srcs_and_dests_node_indices
#         )
#         return ret

#     @staticmethod
#     def backward(ctx, gradout):
#         (
#             row_ptr,
#             col_idx,
#             eids,
#             reltypes,
#             transposed_row_ptr,
#             transposed_col_idx,
#             transposed_eids,
#             transposed_reltypes,
#             message_per_edge,
#             attn_score,
#         ) = ctx.saved_tensors
#         grad_message = th.zeros_like(message_per_edge)
#         grad_attn_score = th.zeros_like(attn_score)
#         K.backward_hgt_full_graph_message_mean_aggregation_csr(
#             outcsr_row_ptr,
#             outcsr_col_idx,
#             outcsr_eids,
#             outcsr_reltypes,
#             edgesoftmax_sum_per_node,
#             normalized_atrtn_score,
#             gradout,
#             grad_message,
#             grad_attn_score,
#         )
#         # fmt: off
#         return None, None, None, None, None, None, None, None, grad_message, grad_attn_score, None
#         # fmt: on


def hgt_full_graph_hetero_attention_ops_coo(
    graph, weight, applied_klinear_node_features, applied_qlinear_node_features
):
    separate_coo_dict = graph.get_separate_coo_original()
    incsr_dict = graph.get_in_csr()

    return HGTFullGraphHeteroAttentionOps.apply(
        incsr_dict["row_ptr"],
        incsr_dict["col_idx"],
        incsr_dict["eids"],
        incsr_dict["rel_types"],
        separate_coo_dict["row_idx"],
        separate_coo_dict["col_idx"],
        separate_coo_dict["eids"],
        separate_coo_dict["rel_ptr"],
        applied_klinear_node_features,
        applied_qlinear_node_features,
        weight,
    )


# def hgt_full_graph_message_mean_aggregation_csr(graph, message_per_edge, attn_score):
#     outcsr_row_ptr = graph["original"]["row_ptr"]
#     outcsr_col_idx = graph["original"]["col_idx"]
#     outcsr_eids = graph["original"]["eids"]
#     outcsr_reltypes = graph["original"]["rel_types"]
#     incsr_row_ptr = graph["transposed"]["row_ptr"]
#     incsr_col_idx = graph["transposed"]["col_idx"]
#     incsr_eids = graph["transposed"]["eids"]
#     incsr_reltypes = graph["transposed"]["rel_types"]
#     ret = th.zeros(
#         graph["original"]["num_nodes"],
#         message_per_edge.size(1),
#         message_per_edge.size(2),
#         dtype=message_per_edge.dtype,
#         device=message_per_edge.device,
#         requires_grad=True,
#     )
#     return HGTFullGraphMessageMeanAggregationCSR.apply(
#         row_ptr,
#         col_idx,
#         eids,
#         reltypes,
#         transposed_row_ptr,
#         transposed_col_idx,
#         transposed_eids,
#         transposed_reltypes,
#         message_per_edge,
#         attn_score,
#         ret,
#     )


class HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR(
    th.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        separate_coo_relptrs,
        separate_coo_row_indices,
        separate_coo_col_indices,
        separate_coo_eids,
        message_generation_weights,
        inputs,
        # relation_pri,
        unnormalized_attn_score,
        edgesoftmax_sum_per_node,
        mu,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
        new_h,
    ):
        K.hgt_full_graph_edge_softmax_ops_separate_coo(
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            separate_coo_relptrs,
            unnormalized_attn_score,
            mu,
            edgesoftmax_sum_per_node,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
        )

        # K.hgt_full_graph_edge_softmax_ops_csr(
        #     incsr_row_ptr,
        #     incsr_col_idx,
        #     incsr_eids,
        #     incsr_reltypes,
        #     unnormalized_attn_score,
        #     mu,
        #     edgesoftmax_sum_per_node,
        #     mu_softmax_applied_unnormalized_attn_score,
        #     normalized_attn_score,
        # )

        K.hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo(
            separate_coo_relptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            inputs,
            message_generation_weights,
            normalized_attn_score,
            # relation_pri,
            # edgesoftmax_sum_per_node,
            # mu,
            new_h,
        )
        # FIXME: why are elements in edgesoftmax_sum_per_node all integers?
        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            separate_coo_relptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            message_generation_weights,
            inputs,
            # relation_pri,
            unnormalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            new_h,
        )
        # print(normalized_attn_score)
        # print(mu)
        return new_h

    # mere fusing of the original rgnn_relational_matmul and hgt_full_graph_edge_softmax_and_mean_aggregation_csr for now
    @staticmethod
    def backward(
        ctx,
        gradout,  # delta out node feature
    ):
        # input_num_head_one_flag = ctx.input_num_head_one_flag

        (
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            separate_coo_relptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            message_generation_weights,
            inputs,
            # relation_pri,
            unnormalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            new_h,
        ) = ctx.saved_tensors

        grad_message_generation_weight = th.zeros_like(
            message_generation_weights, memory_format=th.contiguous_format
        )
        grad_input = th.zeros_like(inputs, memory_format=th.contiguous_format)

        grad_normalized_attn_score = th.zeros_like(
            normalized_attn_score, memory_format=th.contiguous_format
        )

        grad_unnormalized_attn_score = th.zeros_like(
            unnormalized_attn_score, memory_format=th.contiguous_format
        )
        grad_mu = th.zeros_like(mu, memory_format=th.contiguous_format)

        K.backward_hgt_full_graph_fused_message_calc_and_mean_aggregation_separate_coo(
            separate_coo_relptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            inputs,
            th.transpose(message_generation_weights, 2, 3).contiguous(),
            normalized_attn_score,
            new_h,
            grad_input,
            grad_message_generation_weight,
            grad_normalized_attn_score,
            gradout.contiguous(),
        )

        K.backward_hgt_full_graph_enorm_to_unnormalized_attn_score_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            unnormalized_attn_score,
            normalized_attn_score,
            grad_normalized_attn_score,
            mu,
            grad_unnormalized_attn_score,
            grad_mu,
        )
        # print(grad_message_generation_weight)
        # print(grad_input)
        # print(grad_normalized_attn_score)
        # print(grad_unnormalized_attn_score)
        # print(grad_mu)
        # fmt: off
        return None,None,None,None,None,None,None,None,grad_message_generation_weight,grad_input,grad_unnormalized_attn_score,None,grad_mu,None,None,None
        # fmt: on


def hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_csr(
    separate_coo_relptrs,
    separate_coo_row_idx,
    separate_coo_col_idx,
    separate_coo_eids,
    relation_meg_weight,
    inputs,
    graph,
    mu,
    unnormalized_attn_score,
):
    incsr_dict = graph.get_in_csr()
    incsr_row_ptr = incsr_dict["row_ptr"]
    incsr_col_idx = incsr_dict["col_idx"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]

    # TODO: move tensor creation into the kernel
    new_h = th.zeros(
        graph.get_num_nodes(),
        relation_meg_weight.size(1),
        relation_meg_weight.size(3),
        dtype=relation_meg_weight.dtype,
        device=relation_meg_weight.device,
        requires_grad=True,
    ).contiguous()

    edgesoftmax_sum_per_node = th.zeros(
        graph.get_num_nodes(),
        mu.size(1),
        dtype=relation_meg_weight.dtype,
        device=relation_meg_weight.device,
    ).contiguous()

    mu_softmax_applied_unnormalized_attn_score = th.zeros_like(
        unnormalized_attn_score, memory_format=th.contiguous_format
    )
    normalized_attn_score = th.zeros_like(
        unnormalized_attn_score, memory_format=th.contiguous_format
    )

    return HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCSR.apply(
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        separate_coo_relptrs,
        separate_coo_row_idx,
        separate_coo_col_idx,
        separate_coo_eids,
        relation_meg_weight,
        inputs,
        unnormalized_attn_score,
        edgesoftmax_sum_per_node,
        mu,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
        new_h,
    )


class HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        unnormalized_attn_score,
        mu,
        edgesoftmax_sum_per_node,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
        message_per_edge,
        new_h,
    ):

        K.hgt_full_graph_edge_softmax_ops_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            unnormalized_attn_score,
            mu,
            edgesoftmax_sum_per_node,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
        )

        K.hgt_full_graph_message_mean_aggregation_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_reltypes,
            incsr_eids,
            message_per_edge,
            normalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            new_h,
        )

        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            unnormalized_attn_score,
            mu,
            edgesoftmax_sum_per_node,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            message_per_edge,
            new_h,
        )

        return new_h

    @staticmethod
    def backward(ctx, gradout):
        (
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            unnormalized_attn_score,
            mu,
            edgesoftmax_sum_per_node,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            message_per_edge,
            new_h,
        ) = ctx.saved_tensors
        grad_message = th.zeros_like(
            message_per_edge, memory_format=th.contiguous_format
        )

        # FIXME: unique_srcs_and_dests_rel_ptr and unique_srcs_and_dests_node_indices are not used in the backward pass but used in forward pass, check if there is similar issue in the original routine
        K.backward_hgt_full_graph_message_mean_aggregation_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            edgesoftmax_sum_per_node,
            normalized_attn_score,
            gradout,
            grad_message,
        )

        grad_attn_score = th.zeros_like(
            unnormalized_attn_score, memory_format=th.contiguous_format
        )
        grad_mu = th.zeros_like(mu, memory_format=th.contiguous_format)
        K.backward_hgt_full_graph_edge_softmax_ops_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            message_per_edge,
            unnormalized_attn_score,
            normalized_attn_score,
            new_h,
            gradout,
            mu,
            grad_attn_score,
            grad_mu,
        )
        # fmt: off
        return None, None, None, None, None, None, None, None, grad_attn_score, grad_mu, None, None, None, grad_message, None, None, None
        # fmt: on


def hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr(
    graph,
    message_per_edge,
    unnormalized_attn_score,
    mu,
    # edgesoftmax_sum_per_node,
    # mu_softmax_applied_unnormalized_attn_score,
    # normalized_attn_score,
    # unique_srcs_and_dests_rel_ptr,
    # unique_srcs_and_dests_node_indices,
):
    outcsr_dict = graph.get_out_csr()
    incsr_dict = graph.get_in_csr()
    outcsr_row_ptr = outcsr_dict["row_ptr"]
    outcsr_col_idx = outcsr_dict["col_idx"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    incsr_row_ptr = incsr_dict["row_ptr"]
    incsr_col_idx = incsr_dict["col_idx"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]
    mu_softmax_applied_unnormalized_attn_score = th.zeros_like(
        unnormalized_attn_score, memory_format=th.contiguous_format
    )
    normalized_attn_score = th.zeros_like(
        unnormalized_attn_score, memory_format=th.contiguous_format
    )
    edgesoftmax_sum_per_node = th.zeros(
        graph.get_num_nodes(),
        mu.size(1),
        dtype=message_per_edge.dtype,
        device=message_per_edge.device,
    ).contiguous()
    new_h = th.zeros(
        graph.get_num_nodes(),
        message_per_edge.size(1),
        message_per_edge.size(2),
        dtype=message_per_edge.dtype,
        device=message_per_edge.device,
        requires_grad=True,
    ).contiguous()

    # return new_h

    # scale_factor, i.e., sqrt_dk equals math.sqrt(out_dim // n_heads)
    return HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR.apply(
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        unnormalized_attn_score,
        mu,
        edgesoftmax_sum_per_node,
        mu_softmax_applied_unnormalized_attn_score,
        normalized_attn_score,
        message_per_edge,
        # unique_srcs_and_dests_rel_ptr,
        # unique_srcs_and_dests_node_indices,
        new_h,
    )
