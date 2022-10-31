#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from .. import kernels as K


class FusedGat(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
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
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
        )
        # non-tensor arguments should be stored separately see torch repo torch\autograd\function.py
        ctx.slope = slope
        K.fused_gat_kernel_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
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
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
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

        K.backward_fused_gat_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
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
            grad_feat_src,
            grad_el,
            grad_er,
            None,
            None,
            None,
            None,
        )


def fused_gat(graph, feat_src, el, er, slope):
    # g = graph._graph.get_immutable_gidx(utils.to_dgl_context(context(feat_src)))
    incsr_row_ptr = graph["transposed"]["row_ptr"]
    incsr_col_idx = graph["transposed"]["col_idx"]
    incsr_eids = graph["transposed"]["eids"]
    outcsr_row_ptr = graph["original"]["row_ptr"]
    outcsr_col_idx = graph["original"]["col_idx"]
    outcsr_eids = graph["original"]["eids"]

    exp = el.new_empty([incsr_col_idx.numel()] + list(el.size()[1:]))
    s = th.empty_like(el)
    ret = th.empty_like(feat_src)
    return FusedGat.apply(
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
    )
