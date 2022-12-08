#!/usr/bin/env python3
import scipy  # Weird bug in new pytorch when import scipy after import torch
import torch as th

from ..kernels import K


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
        grad_el = th.zeros_like(el, memory_format=th.contiguous_format)
        grad_er = th.zeros_like(er, memory_format=th.contiguous_format)
        grad_feat_src = th.zeros_like(feat_src, memory_format=th.contiguous_format)

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
        # NB: black will format the return statement to a multi-line tuple, but causes error in some cases. However in plain autograd function, packing multiple return values as a tuple is fine. We need to figure out if this is a pytorch issue or ours when we have time.
        # fmt: off
        return None, None, None, None, None, None, None, None, None, None, grad_feat_src, grad_el, grad_er, None, None, None, None,
        # fmt: on


def relational_fused_gat_csr(graph, feat_src, el, er, slope):
    # NB: notice that in rgcn, in-adjacency list is used and therefore, we input the transposed adj list to forward propagation, and the original to backward propagation.
    incsr_dict = graph.get_in_csr()
    outcsr_dict = graph.get_out_csr()
    separate_unique_node_indices = graph.get_separate_unique_node_indices()
    exp = el.new_empty(
        [incsr_dict["col_idx"].numel()] + list(el.size()[1:])
    )  # [num_edges, num_heads]
    s = el.new_empty(
        [graph.get_num_nodes()] + [el.size()[1]]
    )  # [num_dest_nodes, num_heads]
    ret = el.new_empty(
        [graph.get_num_nodes()] + list(feat_src.size()[1:])
    )  # [num_dest_nodes, num_heads, out_feats//num_heads]
    return RelationalFusedGatCSR.apply(
        incsr_dict["row_ptr"],
        incsr_dict["col_idx"],
        incsr_dict["eids"],
        incsr_dict["rel_types"],
        outcsr_dict["row_ptr"],
        outcsr_dict["col_idx"],
        outcsr_dict["eids"],
        outcsr_dict["rel_types"],
        separate_unique_node_indices["rel_ptr"],
        separate_unique_node_indices["node_idx"],
        feat_src,
        el,
        er,
        s,
        exp,
        ret,
        slope,
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
        incsr_reltypes,
        outcsr_row_ptr,
        outcsr_col_idx,
        outcsr_eids,
        outcsr_reltypes,
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
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
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
            incsr_reltypes,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
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
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
            incsr_row_ptr,
            incsr_col_idx,
            incsr_eids,
            incsr_reltypes,
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            feat_compact,
            el_compact,
            er_compact,
            s,
            exp,
            ret,
        ) = ctx.saved_tensors
        slope = ctx.slope
        grad_el_compact = th.zeros_like(el_compact, memory_format=th.contiguous_format)
        grad_er_compact = th.zeros_like(er_compact, memory_format=th.contiguous_format)
        grad_feat_compact = th.zeros_like(
            feat_compact, memory_format=th.contiguous_format
        )

        K.backward_rgat_relational_fused_gat_compact_as_of_node_csr(
            outcsr_row_ptr,
            outcsr_col_idx,
            outcsr_eids,
            outcsr_reltypes,
            unique_srcs_and_dests_rel_ptr,
            unique_srcs_and_dests_node_idx,
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
        # fmt: off
        return None, None, None, None, None, None, None, None, None, None, grad_feat_compact, grad_el_compact, grad_er_compact, None, None, None, None
        # fmt: on


def relational_fused_gat_compact_as_of_node(
    g, feat_compact, el_compact, er_compact, negative_slope
):
    separate_unique_node_indices = g.get_separate_unique_node_indices()
    incsr_dict = g.get_in_csr()
    outcsr_dict = g.get_out_csr()
    exp = el_compact.new_empty(
        [incsr_dict["col_idx"].numel()] + list(el_compact.size()[1:])
    )
    s = el_compact.new_empty([g.get_num_nodes()] + list(el_compact.size()[1:]))
    ret = th.empty(
        [g.get_num_nodes()] + list(feat_compact.size()[1:]),
        dtype=feat_compact.dtype,
        device=feat_compact.device,
        memory_format=th.contiguous_format,
    )
    return RgatRelationalFusedGATCompactAsOfNodeCSR.apply(
        separate_unique_node_indices["rel_ptr"],
        separate_unique_node_indices["node_idx"],
        incsr_dict["row_ptr"],
        incsr_dict["col_idx"],
        incsr_dict["eids"],
        incsr_dict["rel_types"],
        outcsr_dict["row_ptr"],
        outcsr_dict["col_idx"],
        outcsr_dict["eids"],
        outcsr_dict["rel_types"],
        feat_compact,
        el_compact,
        er_compact,
        s,
        exp,
        ret,
        negative_slope,
    )
