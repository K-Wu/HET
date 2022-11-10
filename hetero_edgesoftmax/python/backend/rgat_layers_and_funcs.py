#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class RelationalFusedGatCSR(th.autograd.Function):
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
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
    ):
        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
        )
        ctx.slope = slope
        K.relational_fused_gat_kernel_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
            slope,
        )
        return ret

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
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
        ) = ctx.saved_tensors
        slope = ctx.slope
        grad_el = th.zeros_like(el)
        grad_er = th.zeros_like(er)
        grad_feat_src = th.zeros_like(feat_src)

        K.backward_relational_fused_gat_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
            gradout,
            grad_feat_src,
            grad_el,
            grad_er,
            slope,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_feat_src,
            grad_el,
            grad_er,
            None,
            None,
            None,
            None,
        )


def relational_fused_gat_csr(graph, feat_src, el, er, slope):
    # NB: notice that in rgcn, in-adjacency list is used and therefore, we input the transposed adj list to forward propagation, and the original to backward propagation.

    exp = el.new_empty([graph["transposed"]["col_idx"].numel()] + list(el.size()[1:]))
    s = th.empty_like(el)
    ret = th.empty_like(feat_src)
    return RelationalFusedGatCSR.apply(
        graph["transposed"]["row_ptr"],
        graph["transposed"]["col_idx"],
        graph["transposed"]["eids"],
        graph["transposed"]["rel_types"],
        graph["original"]["row_ptr"],
        graph["original"]["col_idx"],
        graph["original"]["eids"],
        graph["original"]["rel_types"],
        graph["separate"]["unique_srcs_and_dests"]["rel_ptr"],
        graph["separate"]["unique_srcs_and_dests"]["unique_node_idxes"],
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
    )


class RgatRelationalMatmul(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_relptrs,
        separate_coo_node_indicies,
        separate_coo_eids,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        weights,
        input,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            weights,
            input,
        )
        K.rgat_relational_matmul(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            weights,
            input,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            weights,
            input,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weights)
        grad_input = th.zeros_like(input)
        K.rgat_relational_matmul_backward(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_indices,
            th.transpose(weights, 2, 3),
            input,
            gradout,
            grad_input,
            grad_weight,
        )
        return (
            None,
            None,
            None,
            grad_input,
            grad_weight,
            None,
        )


def rgat_relational_matmul(
    separate_coo_relptrs,
    separate_coo_node_indicies,
    separate_coo_eids,
    unique_srcs_and_dests_rel_ptr,
    unique_srcs_and_dests_node_indices,
    weights,
    input,
):
    ret = th.zeros(
        (separate_coo_node_indicies.numel(), weights.size(1), weights.size(3)),
        dtype=weights.dtype,
        device=weights.device,
        requires_grad=True,
    )
    return RgatRelationalMatmul.apply(
        separate_coo_relptrs,
        separate_coo_node_indicies,
        separate_coo_eids,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        weights,
        input,
        ret,
    )


class RgatRelationalFusedGATCompactAsOfNodeCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_idx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        feat_compact,
        el_compact,
        er_compact,
        s,
        exp,
        ret,
        slope,
    ):
        ctx.save_for_backward(
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            feat_compact,
            el_compact,
            er_compact,
            s,
            exp,
            ret,
        )
        ctx.slope = slope
        K.rgat_relational_fused_gat_compact_as_of_node_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            feat_compact,
            el_compact,
            er_compact,
            s,
            exp,
            ret,
            slope,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            feat_compact,
            el_compact,
            er_compact,
            s,
            exp,
            ret,
        ) = ctx.saved_tensors
        slope = ctx.slope
        grad_el_compact = th.zeros_like(el_compact)
        grad_er_compact = th.zeros_like(er_compact)
        grad_feat_compact = th.zeros_like(feat_compact)

        K.backward_rgat_relational_fused_gat_compact_as_of_node_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            feat_compact,
            el_compact,
            er_compact,
            s,
            exp,
            ret,
            gradout,
            grad_feat_compact,
            grad_el_compact,
            grad_er_compact,
            slope,
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_feat_compact,
            grad_el_compact,
            grad_er_compact,
            None,
            None,
            None,
            None,
        )


def relational_fused_gat_compact_as_of_node(
    g, feat_compact, el_compact, er_compact, negative_slope
):
    exp = el_compact.new_empty(
        [g["transposed"]["col_idx"].numel()] + list(el_compact.size()[1:])
    )
    s = el_compact.new_empty([g.get_num_of_nodes()] + list(el_compact.size()[1:]))
    ret = th.empty(
        [g.get_num_of_nodes()] + list(feat_compact.size()[1:]),
        dtype=feat_compact.dtype,
        device=feat_compact.device,
    )
    return RgatRelationalFusedGATCompactAsOfNodeCSR.apply(
        g["separate"]["unique_srcs_and_dests"]["rel_ptr"],
        g["separate"]["unique_srcs_and_dests"]["unique_node_idxes"],
        g["transposed"]["csr"]["row_ptr"],
        g["transposed"]["csr"]["col_idx"],
        g["transposed"]["csr"]["eids"],
        g["original"]["csr"]["row_ptr"],
        g["original"]["csr"]["col_idx"],
        g["original"]["csr"]["eids"],
        feat_compact,
        el_compact,
        er_compact,
        s,
        exp,
        ret,
        negative_slope,
    )


class RgatRelationalMatmulCompactAsOfNode(th.autograd.Function):
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
        K.rgat_relational_matmul_compact_as_of_node(
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
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weight)
        grad_node_feat = th.zeros_like(node_feat)
        K.backward_rgat_relational_matmul_compact_as_of_node(
            separate_coo_relptrs,
            separate_coo_node_indices,
            separate_coo_eids,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            weight,
            node_feat,
            ret,
            gradout,
            grad_weight,
            grad_node_feat,
        )
        return (
            None,
            None,
            grad_weight,
            grad_node_feat,
            None,
        )


def rgat_relational_matmul_compact_as_of_node(
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
    return RgatRelationalMatmulCompactAsOfNode.apply(
        unique_srcs_and_dests_rel_ptr,
        unique_srcs_and_dests_node_indices,
        weight,
        node_feat,
        ret,
    )
