#!/usr/bin/env python3
import torch


def fused_gat_kernel_csr(row_ptr, col_idx, eids, feat_src, el, er, s, exp, ret, slope):
    torch.ops.torch_hetero_edgesoftmax.fused_gat_kernel_csr(
        row_ptr, col_idx, eids, feat_src, el, er, s, exp, ret, slope
    )


def backward_fused_gat_csr(
    row_ptr,
    col_idx,
    eids,
    feat_src,
    el,
    er,
    s,
    exp,
    ret,
    grad_out,
    grad_feat_src,
    grad_el,
    grad_er,
    slope,
):
    torch.ops.torch_hetero_edgesoftmax.backward_fused_gat_csr(
        row_ptr,
        col_idx,
        eids,
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        grad_out,
        grad_feat_src,
        grad_el,
        grad_er,
        slope,
    )
