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

    exp = el.new_empty([incsr_col_idx.numel()] + list(el.size()[1:]))
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
        weights,
        input,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            weights,
            input,
        )
        K.rgat_relational_matmul(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
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
            weights,
            input,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weight)
        grad_input = th.zeros_like(input)
        K.rgat_relational_matmul_backward(
            separate_coo_relptrs,
            separate_coo_node_indicies,
            separate_coo_eids,
            weights,
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
    separate_coo_relptrs, separate_coo_node_indicies, separate_coo_eids, weights, input
):
    ret = th.zeros(
        (graph.get_num_edges(), weight.size(1), weight.size(3)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgatRelationalMatmul.apply(
        separate_coo_relptrs,
        separate_coo_node_indicies,
        separate_coo_eids,
        weights,
        input,
        ret,
    )
