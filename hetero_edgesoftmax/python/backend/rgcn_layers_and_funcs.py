#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class RgcnFirstLayerCSR(th.autograd.Function):
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
        weight,
        norm,
        ret,
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
            weight,
            norm,
        )
        K.rgcn_layer0_csr(
            incsr_row_ptr, incsr_col_idx, incsr_eids, incsr_reltypes, weight, norm, ret
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
            weight,
            norm,
        ) = ctx.saved_tensors
        print(weight.numel())
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer0_backward_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            gradout,
            norm,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_weight, None, None


def rgcn_layer0_csr(graph, weight, norm):
    # NB: notice that in rgcn, in-adjacency list is used and therefore, we input the transposed adj list to forward propagation, and the original to backward propagation.
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    incsr_row_ptr = incsr_dict["row_ptr"]
    incsr_col_idx = incsr_dict["col_idx"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]
    outcsr_row_ptr = outcsr_dict["row_ptr"]
    outcsr_col_idx = outcsr_dict["col_idx"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    ret = th.zeros(
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnFirstLayerCSR.apply(
        incsr_row_ptr,
        incsr_col_idx,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        weight,
        norm,
        ret,
    )


class RgcnSecondLayerCSR(th.autograd.Function):
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
        x,
        weight,
        norm,
        ret,
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
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_csr(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            x,
            weight,
            norm,
            ret,
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
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


class RgcnSecondLayerCSRHybridAssign(th.autograd.Function):
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
        x,
        weight,
        norm,
        ret,
        num_blocks_on_node_forward,
        num_blocks_on_node_backward,
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
            weight,
            norm,
            x,
        )
        ctx.num_blocks_on_node_backward = num_blocks_on_node_backward
        K.rgcn_layer1_csr_hybrid_assign(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            x,
            weight,
            norm,
            ret,
            num_blocks_on_node_forward,
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
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        num_blocks_on_node_backward = ctx.num_blocks_on_node_backward
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_csr_hybrid_assign(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
            num_blocks_on_node_backward,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


class RgcnLayer1SeparateCoo(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_rel_ptr,
        separate_coo_eids,
        separate_coo_row_idx,
        separate_coo_col_idx,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_idx,
            separate_coo_col_idx,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_separate_coo(
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_idx,
            separate_coo_col_idx,
            x,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_rel_ptr,
            separate_coo_eids,
            separate_coo_row_idx,
            separate_coo_col_idx,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


def rgcn_layer1_separate_coo(
    separate_coo_rel_ptr,
    separate_coo_eids,
    separate_coo_row_idx,
    separate_coo_col_idx,
    graph,
    x,
    weight,
    norm,
):
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    incsr_row_ptr = incsr_dict["row_ptr"]
    ret = th.zeros(
        (incsr_row_ptr.numel() - 1, weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    outcsr_row_ptr = outcsr_dict["row_ptr"]
    outcsr_col_idx = outcsr_dict["col_idx"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    return RgcnLayer1SeparateCoo.apply(
        separate_coo_rel_ptr,
        separate_coo_eids,
        separate_coo_row_idx,
        separate_coo_col_idx,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
        x,
        weight,
        norm,
        ret,
    )


def rgcn_layer1_csr(
    graph,
    x,
    weight,
    norm,
    hybrid_assign_flag=False,
    num_blocks_on_node_forward=None,
    num_blocks_on_node_backward=None,
):
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    incsr_row_ptr = incsr_dict["row_ptr"]
    incsr_col_idx = incsr_dict["col_idx"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]
    outcsr_row_ptr = outcsr_dict["row_ptr"]
    outcsr_col_idx = outcsr_dict["col_idx"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    ret = th.zeros(
        (incsr_row_ptr.numel() - 1, weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    if hybrid_assign_flag:
        return RgcnSecondLayerCSRHybridAssign.apply(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            ret,
            num_blocks_on_node_forward,
            num_blocks_on_node_backward,
        )

    else:
        return RgcnSecondLayerCSR.apply(
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            ret,
        )
