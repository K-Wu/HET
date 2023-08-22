#!/usr/bin/env python3
# From seastar-paper-version/exp/rgcn/seastar/train.py
# the paper copy of seastar can be obtained from www.cse.cuhk.edu.hk/~jcheng/papers/seastar_eurosys21.pdf
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


class SeastarRgcnSecondLayerCOO(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        in_row_indices,
        in_col_indices,
        in_eids,
        in_reltypes,
        out_row_indices,
        out_col_indices,
        out_eids,
        out_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            in_row_indices,
            in_col_indices,
            in_eids,
            in_reltypes,
            out_row_indices,
            out_col_indices,
            out_eids,
            out_reltypes,
            weight,
            norm,
            x,
        )
        K.seastar_rgcn_layer1_coo(
            in_row_indices,
            in_col_indices,
            in_eids,
            in_reltypes,
            x,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            in_row_indices,
            in_col_indices,
            in_eids,
            in_reltypes,
            out_row_indices,
            out_col_indices,
            out_eids,
            out_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x, memory_format=th.contiguous_format)
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        K.seastar_backward_rgcn_layer1_coo(
            out_row_indices,
            out_col_indices,
            out_eids,
            out_reltypes,
            x,
            weight,
            norm,
            gradout,
            grad_x,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_x, None, None, None


def seastar_rgcn_layer1_coo(graph, x, weight, norm):
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    in_row_indices = incsr_dict["row_indices"]
    in_col_indices = incsr_dict["col_indices"]
    in_eids = incsr_dict["eids"]
    in_reltypes = incsr_dict["rel_types"]
    out_row_indices = outcsr_dict["row_indices"]
    out_col_indices = outcsr_dict["col_indices"]
    out_eids = outcsr_dict["eids"]
    transposed_reltypes = outcsr_dict["rel_types"]
    ret = th.zeros(
        (graph.get_num_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    ).contiguous()
    return SeastarRgcnSecondLayerCOO.apply(
        in_row_indices,
        in_col_indices,
        in_eids,
        in_reltypes,
        out_row_indices,
        out_col_indices,
        out_eids,
        transposed_reltypes,
        x,
        weight,
        norm,
        ret,
    )


class SeastarRgcnFirstLayerCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_indices,
        outcsr_eids,
        outcsr_reltypes,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
        )
        K.seastar_rgcn_layer0_csr(
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
        ) = ctx.saved_tensors
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        K.seastar_backward_rgcn_layer0_csr(
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            gradout,
            norm,
            grad_weight,
        )
        return None, None, None, None, None, None, None, None, grad_weight, None, None


def seastar_rgcn_layer0_csr(graph, weight, norm):
    # NB: notice that in rgcn, in-adjacency list is used and therefore, we input the transposed adj list to forward propagation, and the original to backward propagation.
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    incsr_row_ptr = incsr_dict["row_ptrs"]
    incsr_col_indices = incsr_dict["col_indices"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]
    outcsr_row_ptr = outcsr_dict["row_ptrs"]
    outcsr_col_indices = outcsr_dict["col_indices"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    ret = th.zeros(
        (weight.size(1), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    )
    return SeastarRgcnFirstLayerCSR.apply(
        incsr_row_ptr,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_indices,
        outcsr_eids,
        outcsr_reltypes,
        weight,
        norm,
        ret,
    )


class SeastarRgcnSecondLayerCSR(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_indices,
        outcsr_eids,
        outcsr_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        )
        K.seastar_rgcn_layer1_csr(
            incsr_row_ptr,
            incsr_col_indices,
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
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x, memory_format=th.contiguous_format)
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        K.seastar_backward_rgcn_layer1_csr(
            outcsr_row_ptr,
            outcsr_col_indices,
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


def seastar_rgcn_layer1_csr(
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
    incsr_row_ptr = incsr_dict["row_ptrs"]
    incsr_col_indices = incsr_dict["col_indices"]
    incsr_eids = incsr_dict["eids"]
    incsr_reltypes = incsr_dict["rel_types"]
    outcsr_row_ptr = outcsr_dict["row_ptrs"]
    outcsr_col_indices = outcsr_dict["col_indices"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    ret = th.zeros(
        (graph.get_num_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    ).contiguous()
    if hybrid_assign_flag:
        return SeastarRgcnSecondLayerCSRHybridAssign.apply(
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
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
        return SeastarRgcnSecondLayerCSR.apply(
            incsr_row_ptr,
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            x,
            weight,
            norm,
            ret,
        )


class SeastarRgcnSecondLayerCSRHybridAssign(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        incsr_row_ptr,
        incsr_col_indices,
        incsr_eids,
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_indices,
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
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        )
        ctx.num_blocks_on_node_backward = num_blocks_on_node_backward
        K.seastar_rgcn_layer1_csr_hybrid_assign(
            incsr_row_ptr,
            incsr_col_indices,
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
            incsr_col_indices,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        num_blocks_on_node_backward = ctx.num_blocks_on_node_backward
        grad_x = th.zeros_like(x, memory_format=th.contiguous_format)
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        K.seastar_backward_rgcn_layer1_csr_hybrid_assign(
            outcsr_row_ptr,
            outcsr_col_indices,
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
        separate_coo_rel_ptrs,
        separate_coo_eids,
        separate_coo_row_indices,
        separate_coo_col_indices,
        outcsr_row_ptr,
        outcsr_col_indices,
        outcsr_eids,
        outcsr_reltypes,
        x,
        weight,
        norm,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        )
        K.rgcn_layer1_separate_coo(
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            x,
            weight,
            norm,
            ret,
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            outcsr_row_ptr,
            outcsr_col_indices,
            outcsr_eids,
            outcsr_reltypes,
            weight,
            norm,
            x,
        ) = ctx.saved_tensors
        grad_x = th.zeros_like(x, memory_format=th.contiguous_format)
        grad_weight = th.zeros_like(weight, memory_format=th.contiguous_format)
        grad_norm = th.zeros_like(norm, memory_format=th.contiguous_format)
        K.backward_rgcn_layer1_separate_coo(
            separate_coo_rel_ptrs,
            separate_coo_eids,
            separate_coo_row_indices,
            separate_coo_col_indices,
            x,
            th.transpose(weight, 1, 2).contiguous(),
            norm,
            grad_norm,
            grad_x,
            gradout,
            grad_weight,
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
            grad_x,
            grad_weight,
            grad_norm,
            None,
        )


def rgcn_layer1_separate_coo(
    graph,
    x,
    weight,
    norm,
):
    separate_coo_original_dict = graph.get_separate_coo_original()
    outcsr_dict = graph.get_out_csr()
    ret = th.zeros(
        (graph.get_num_nodes(), weight.size(2)),
        dtype=weight.dtype,
        device=weight.device,
        requires_grad=True,
    ).contiguous()
    outcsr_row_ptr = outcsr_dict["row_ptrs"]
    outcsr_col_indices = outcsr_dict["col_indices"]
    outcsr_eids = outcsr_dict["eids"]
    outcsr_reltypes = outcsr_dict["rel_types"]
    return RgcnLayer1SeparateCoo.apply(
        separate_coo_original_dict["rel_ptrs"],
        separate_coo_original_dict["eids"],
        separate_coo_original_dict["row_indices"],
        separate_coo_original_dict["col_indices"],
        outcsr_row_ptr,
        outcsr_col_indices,
        outcsr_eids,
        outcsr_reltypes,
        x,
        weight,
        norm,
        ret,
    )


class RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_indices,
        separate_coo_col_indices,
        separate_unique_node_indices_rel_ptrs,
        separate_unique_node_indices_node_indices,
        feat_src,
        enorm,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_unique_node_indices_rel_ptrs,
            separate_unique_node_indices_node_indices,
            feat_src,
            enorm,
            ret,
        )
        K.rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            {
                "rel_ptrs_row": separate_unique_node_indices_rel_ptrs,
                "node_indices_row": separate_unique_node_indices_node_indices,
            },
            feat_src,
            enorm,
            ret,
            False,  # compact_direct_indexing_flag
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            separate_unique_node_indices_rel_ptrs,
            separate_unique_node_indices_node_indices,
            feat_src,
            enorm,
            ret,
        ) = ctx.saved_tensors
        grad_feat_src = th.zeros_like(
            feat_src, memory_format=th.contiguous_format)

        K.backward_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            {
                "rel_ptrs_row": separate_unique_node_indices_rel_ptrs,
                "node_indices_row": separate_unique_node_indices_node_indices,
            },
            feat_src,
            enorm,
            ret,
            gradout,
            grad_feat_src,
            False,  # compact_direct_indexing_flag
        )
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return None, None, None, None, None, None, grad_feat_src, None, None
        # fmt: on


class RGCNNodeMeanAggregationCompactAsOfNodeDirectIndexingSeparateCOO(
    th.autograd.Function
):
    @staticmethod
    def forward(
        ctx,
        separate_coo_eids,
        separate_coo_rel_ptrs,
        separate_coo_row_indices,
        separate_coo_col_indices,
        inverse_indices_row,
        feat_src,
        enorm,
        ret,
    ):
        ctx.save_for_backward(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            inverse_indices_row,
            feat_src,
            enorm,
            ret,
        )
        K.rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            {"inverse_indices_row": inverse_indices_row},
            feat_src,
            enorm,
            ret,
            True,  # compact_direct_indexing_flag
        )
        return ret

    @staticmethod
    def backward(ctx, gradout):
        (
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            inverse_indices_row,
            feat_src,
            enorm,
            ret,
        ) = ctx.saved_tensors
        grad_feat_src = th.zeros_like(
            feat_src, memory_format=th.contiguous_format)

        K.backward_rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(
            separate_coo_eids,
            separate_coo_rel_ptrs,
            separate_coo_row_indices,
            separate_coo_col_indices,
            {"inverse_indices_row": inverse_indices_row},
            feat_src,
            enorm,
            ret,
            gradout,
            grad_feat_src,
            True,  # compact_direct_indexing_flag
        )
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return None, None, None, None, None, grad_feat_src, None, None
        # fmt: on


def rgcn_node_mean_aggregation_compact_as_of_node_separate_coo(g, feat_compact, enorm):
    separate_coo_dict = g.get_separate_coo_original()
    separate_unique_node_indices = g.get_separate_unique_node_indices()

    ret = th.empty(
        [g.get_num_nodes()] + list(feat_compact.size()[1:]),
        dtype=feat_compact.dtype,
        device=feat_compact.device,
        memory_format=th.contiguous_format,
    )
    return RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO.apply(
        separate_coo_dict["eids"],
        separate_coo_dict["rel_ptrs"],
        separate_coo_dict["row_indices"],
        separate_coo_dict["col_indices"],
        separate_unique_node_indices["rel_ptrs"],
        separate_unique_node_indices["node_indices"],
        feat_compact,
        enorm,
        ret,
    )


def rgcn_node_mean_aggregation_compact_as_of_node_separate_coo_single_sided(
    g, feat_compact_src, enorm, compact_direct_indexing_flag
):
    separate_coo_dict = g.get_separate_coo_original()
    separate_unique_node_indices_single_sided = (
        g.get_separate_unique_node_indices_single_sided()
    )

    ret = th.empty(
        [g.get_num_nodes()] + list(feat_compact_src.size()[1:]),
        dtype=feat_compact_src.dtype,
        device=feat_compact_src.device,
        memory_format=th.contiguous_format,
    )
    if compact_direct_indexing_flag:
        separate_unique_node_indices_dict_single_sided_inverse_idx = (
            g.get_separate_unique_node_indices_single_sided_inverse_idx()
        )
        return RGCNNodeMeanAggregationCompactAsOfNodeDirectIndexingSeparateCOO.apply(
            separate_coo_dict["eids"],
            separate_coo_dict["rel_ptrs"],
            separate_coo_dict["row_indices"],
            separate_coo_dict["col_indices"],
            separate_unique_node_indices_dict_single_sided_inverse_idx[
                "inverse_indices_row"
            ],
            feat_compact_src,
            enorm,
            ret,
        )

    else:
        return RGCNNodeMeanAggregationCompactAsOfNodeSeparateCOO.apply(
            separate_coo_dict["eids"],
            separate_coo_dict["rel_ptrs"],
            separate_coo_dict["row_indices"],
            separate_coo_dict["col_indices"],
            separate_unique_node_indices_single_sided["rel_ptrs_row"],
            separate_unique_node_indices_single_sided["node_indices_row"],
            feat_compact_src,
            enorm,
            ret,
        )
