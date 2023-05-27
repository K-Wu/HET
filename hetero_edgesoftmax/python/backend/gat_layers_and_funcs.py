#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class FusedGat(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptrs,
        incsr_col_indices,
        incsr_eids,
        outcsr_row_ptrs,
        outcsr_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            outcsr_row_ptrs,
            outcsr_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
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
            incsr_row_ptrs,
            incsr_col_indices,
            incsr_eids,
            outcsr_row_ptrs,
            outcsr_col_indices,
            outcsr_eids,
            feat_src,
            el,
            er,
            s,
            exp,
            ret,
        ) = ctx.saved_tensors
        slope = ctx.slope
        grad_el = th.zeros_like(el, memory_format=th.contiguous_format)
        grad_er = th.zeros_like(er, memory_format=th.contiguous_format)
        grad_feat_src = th.zeros_like(feat_src, memory_format=th.contiguous_format)

        K.backward_fused_gat_csr(
            outcsr_row_ptrs,
            outcsr_col_indices,
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
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    exp = el.new_empty([incsr_dict["col_indices"].numel()] + list(el.size()[1:]))
    s = th.empty_like(el, memory_format=th.contiguous_format)
    ret = th.empty_like(feat_src, memory_format=th.contiguous_format)
    return FusedGat.apply(
        incsr_dict["row_ptrs"],
        incsr_dict["col_indices"],
        incsr_dict["eids"],
        outcsr_dict["row_ptrs"],
        outcsr_dict["col_indices"],
        outcsr_dict["eids"],
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
    )
