#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


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
        grad_el = th.zeros_like(el, memory_format=th.contiguous_format)
        grad_er = th.zeros_like(er, memory_format=th.contiguous_format)
        grad_feat_src = th.zeros_like(feat_src, memory_format=th.contiguous_format)

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
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    exp = el.new_empty([incsr_dict["col_idx"].numel()] + list(el.size()[1:]))
    s = th.empty_like(el, memory_format=th.contiguous_format)
    ret = th.empty_like(feat_src, memory_format=th.contiguous_format)
    return FusedGat.apply(
        incsr_dict["row_ptr"],
        incsr_dict["col_idx"],
        incsr_dict["eids"],
        outcsr_dict["row_ptr"],
        outcsr_dict["col_idx"],
        outcsr_dict["eids"],
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
    )
