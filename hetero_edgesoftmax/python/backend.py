# From seastar-paper-version\exp\rgcn\seastar\train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from . import kernels as K


class RgcnFirstLayer(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, weight, norm, ret):
        ctx.save_for_backward(graph, weight.size(), norm)
        K.rgcn_layer0(graph, weight, norm, ret)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        graph, weight_size, norm = ctx.saved_tensors
        grad_weight = th.zeros(weight_size, dtype=norm.dtype, device=norm.device)
        K.rgcn_layer0_backward(graph, gradout, norm, grad_weight)
        return None, grad_weight, None, None


def rgcn_layer0(graph, weight, norm):
    g = graph._graph
    ret = th.zeros(
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnFirstLayer.apply(g, weight, norm, ret)


class RgcnSecondLayer(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, x, weight, norm, ret):
        ctx.save_for_backward(graph, weight, norm, x)
        K.rgcn_layer1(graph, x, weight, norm, ret)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        graph, weight, norm, x = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward(graph, x, weight, norm, gradout, grad_x, grad_weight)
        return None, grad_x, None, None, None


def rgcn_layer1(graph, x, weight, norm):
    g = graph._graph
    ret = th.zeros(
        (graph.number_of_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnSecondLayer.apply(g, x, weight, norm, ret)


class RgcnFirstLayerCSR(th.autograd.Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, eids, reltypes, weight, norm, ret):

        ctx.save_for_backward(row_ptr, col_idx, eids, reltypes, weight, norm)
        K.rgcn_layer0_csr(row_ptr, col_idx, eids, reltypes, weight, norm, ret)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        row_ptr, col_idx, eids, reltypes, weight, norm = ctx.saved_tensors
        print(weight.numel())
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer0_backward_csr(
            row_ptr, col_idx, eids, reltypes, gradout, norm, grad_weight
        )
        return None, None, None, None, grad_weight, None, None


def rgcn_layer0_csr(graph, weight, norm):
    row_ptr = graph["original"]["row_ptr"]
    col_idx = graph["original"]["col_idx"]
    eids = graph["original"]["eids"]
    reltypes = graph["original"]["rel_types"]
    ret = th.zeros(
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnFirstLayerCSR.apply(row_ptr, col_idx, eids, reltypes, weight, norm, ret)


class RgcnSecondLayerCSR(th.autograd.Function):
    @staticmethod
    def forward(ctx, row_ptr, col_idx, eids, reltypes, x, weight, norm, ret):
        ctx.save_for_backward(row_ptr, col_idx, eids, reltypes, weight, norm, x)
        K.rgcn_layer1_csr(row_ptr, col_idx, eids, reltypes, x, weight, norm, ret)
        return ret

    @staticmethod
    def backward(ctx, gradout):
        row_ptr, col_idx, eids, reltypes, weight, norm, x = ctx.saved_tensors
        grad_x = th.zeros_like(x)
        grad_weight = th.zeros_like(weight)
        K.rgcn_layer1_backward_csr(
            row_ptr,
            col_idx,
            eids,
            reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, grad_x, None, None, None


def rgcn_layer1_csr(graph, x, weight, norm):
    row_ptr = graph["transposed"]["row_ptr"]
    col_idx = graph["transposed"]["col_idx"]
    eids = graph["transposed"]["eids"]
    reltypes = graph["transposed"]["rel_types"]
    ret = th.zeros(
        (row_ptr.numel() - 1, weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return RgcnSecondLayerCSR.apply(
        row_ptr, col_idx, eids, reltypes, x, weight, norm, ret
    )
