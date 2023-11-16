#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

# import nvtx
from ..kernels import K


class HGTFullGraphHeteroAttentionOps(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptrs,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        separate_coo_row_indices,
        separate_coo_col_indices,
        separate_coo_eids,
        separate_coo_relptrs,
        applied_klinear_node_features,
        applied_qlinear_node_features,
        attn_score_weight,
    ):
        unnormalized_attn_score = th.zeros(
            (
                separate_coo_row_indices.numel(),
                attn_score_weight.size(1),
            ),  # weight size (self.num_relations, num_heads, self.d_k, self.d_k)
            dtype=attn_score_weight.dtype,
            device=attn_score_weight.device,
            requires_grad=True,
        )

        attn_score_inner_product = th.zeros(
            separate_coo_eids.size(0),
            applied_klinear_node_features.size(1),
            applied_klinear_node_features.size(2),
            dtype=unnormalized_attn_score.dtype,
            device=unnormalized_attn_score.device,
        ).contiguous()
        K.hgt_full_graph_hetero_attention_ops_coo(
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            separate_coo_relptrs,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_weight,
            attn_score_inner_product,
            unnormalized_attn_score,
        )

        ctx.save_for_backward(
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_indices,
            separate_coo_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            separate_coo_relptrs,
            applied_klinear_node_features,
            applied_qlinear_node_features,
            attn_score_inner_product,
            attn_score_weight,
            unnormalized_attn_score,
        ) = ctx.saved_tensors
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            separate_coo_row_indices,
            separate_coo_col_indices,
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
        # print("grad_k", grad_k)
        # print("grad_q", grad_q)
        # print("grad_attn_weight", grad_attn_weight)
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return  None, None, None, None, None, None, None, None,  grad_k, grad_q, grad_attn_weight
        # fmt: on


class HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCOO(
    th.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptrs,
        incsr_col_indices,
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
        # with nvtx.annotate("hector_op_category = edge softmax", color="cyan"):
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

        # with nvtx.annotate("hector_op_category = mm + weighted aggregation", color="cyan"):
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            separate_coo_relptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_coo_eids,
            message_generation_weights,
            inputs,
            unnormalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            new_h,
        )
        return new_h

    # mere fusing of the original rgnn_relational_matmul and hgt_full_graph_edge_softmax_and_mean_aggregation_csr for now
    @staticmethod
    def backward(
        ctx,
        gradout,  # delta out node feature
    ):
        (
            incsr_row_ptrs,
            incsr_col_indices,
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

        # expose the unused csr as a choice
        # TODO: rename this function, this class and the function calling this class as SeparateCOO
        if 0:
            K.backward_hgt_full_graph_enorm_to_unnormalized_attn_score_csr(
                incsr_row_ptrs,
                incsr_col_indices,
                incsr_eids,
                incsr_reltypes,
                unnormalized_attn_score,
                normalized_attn_score,
                grad_normalized_attn_score,
                mu,
                grad_unnormalized_attn_score,
                grad_mu,
            )
        else:
            num_nodes = incsr_row_ptrs.shape[0] - 1
            num_heads = unnormalized_attn_score.shape[1]
            sum_incoming_edges_product_softmax_score = th.zeros(
                [num_nodes, num_heads],
                dtype=normalized_attn_score.dtype,
                device=incsr_row_ptrs.device,
            ).contiguous()
            K.backward_hgt_full_graph_enorm_to_unnormalized_attn_score_separate_coo(
                separate_coo_row_indices,
                separate_coo_col_indices,
                separate_coo_eids,
                separate_coo_relptrs,
                unnormalized_attn_score,
                normalized_attn_score,
                grad_normalized_attn_score,
                mu,
                grad_unnormalized_attn_score,
                grad_mu,
                sum_incoming_edges_product_softmax_score,
            )
        # print("grad_mu", grad_mu)
        # print("_grad_normalized_attn_score", grad_normalized_attn_score)
        # print("grad_unnormalized_attn_score", grad_unnormalized_attn_score)
        # print("grad_input", grad_input)
        # print("grad_message_generation_weight", grad_message_generation_weight)
        # print("gradout", gradout)
        # fmt: off
        return None,None,None,None,None,None,None,None,grad_message_generation_weight,grad_input,grad_unnormalized_attn_score,None,grad_mu,None,None,None
        # fmt: on


class HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR(
    th.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptrs,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptrs,
        outcsr_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            unnormalized_attn_score,
            mu,
            edgesoftmax_sum_per_node,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
        )

        K.hgt_full_graph_message_mean_aggregation_csr(
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_reltypes,
            incsr_eids,
            message_per_edge,
            normalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            new_h,
        )

        ctx.save_for_backward(
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptrs,
            outcsr_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptrs,
            outcsr_col_indices,
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

        K.backward_hgt_full_graph_message_mean_aggregation_csr(
            outcsr_row_ptrs,
            outcsr_col_indices,
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
            outcsr_row_ptrs,
            outcsr_col_indices,
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
        # print("grad_mu", grad_mu)
        # print("grad_attn_score", grad_attn_score)
        # print("grad_message", grad_message)

        # fmt: off
        return None, None, None, None, None, None, None, None, grad_attn_score, grad_mu, None, None, None, grad_message, None, None, None
        # fmt: on


def hgt_full_graph_hetero_attention_ops_coo(
    graph, weight, applied_klinear_node_features, applied_qlinear_node_features
):
    separate_coo_dict = graph.get_separate_coo_original()
    incsr_dict = graph.get_in_csr()

    return HGTFullGraphHeteroAttentionOps.apply(
        incsr_dict["row_ptrs"],
        incsr_dict["col_indices"],
        incsr_dict["eids"],
        incsr_dict["rel_types"],
        separate_coo_dict["row_indices"],
        separate_coo_dict["col_indices"],
        separate_coo_dict["eids"],
        separate_coo_dict["rel_ptrs"],
        applied_klinear_node_features,
        applied_qlinear_node_features,
        weight,
    )


def hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_coo(
    relation_meg_weight,
    inputs,
    graph,
    mu,
    unnormalized_attn_score,
):
    separate_coo_original_dict = graph.get_separate_coo_original()
    incsr_dict = graph.get_in_csr()
    incsr_row_ptrs = incsr_dict["row_ptrs"]
    incsr_col_indices = incsr_dict["col_indices"]
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

    return (
        HGTFullGraphMessageCalcEdgeSoftmaxAndMessageMeanAggregationCOO.apply(
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            separate_coo_original_dict["rel_ptrs"],
            separate_coo_original_dict["row_indices"],
            separate_coo_original_dict["col_indices"],
            separate_coo_original_dict["eids"],
            relation_meg_weight,
            inputs,
            unnormalized_attn_score,
            edgesoftmax_sum_per_node,
            mu,
            mu_softmax_applied_unnormalized_attn_score,
            normalized_attn_score,
            new_h,
        )
    )


def hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr(
    graph,
    message_per_edge,
    unnormalized_attn_score,
    mu,
):
    outcsr_dict = graph.get_out_csr()
    incsr_dict = graph.get_in_csr()
    outcsr_row_ptrs = outcsr_dict["row_ptrs"]
    outcsr_col_indices = outcsr_dict["col_indices"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    incsr_row_ptrs = incsr_dict["row_ptrs"]
    incsr_col_indices = incsr_dict["col_indices"]
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

    # scale_factor, i.e., sqrt_dk equals math.sqrt(out_dim // num_heads)
    return HGTFullGraphEdgeSoftmaxAndMessageMeanAggregationOpsCSR.apply(
        incsr_row_ptrs,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptrs,
        outcsr_col_indices,
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
