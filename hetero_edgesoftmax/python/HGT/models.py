#!/usr/bin/env python3
# from graphiler/examples/HGT/HGT_DGL.py
import math

import torch
import torch.nn as nn
import torch.jit

# import dgl.function as fn
# from dgl.nn.functional import edge_softmax
from .. import backend as B
from .. import utils_lite


class HET_HGTLayerHetero(nn.Module):
    @utils_lite.warn_default_arguments
    def __init__(
        self,
        in_dim,
        out_dim,
        mydglgraph,
        src_node_type_per_canonical_edge_type,
        dst_node_type_per_canonical_edge_type,
        n_heads=1,
        dropout=0.2,
        use_norm=False,
        multiply_among_weights_first_flag=False,
        hgt_fused_attn_score_flag=False,
        compact_as_of_node_flag=False,
        fused_message_mean_aggregation_flag=False,
    ):
        super(HET_HGTLayerHetero, self).__init__()
        self.hgt_fused_attn_score_flag = hgt_fused_attn_score_flag
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.fused_message_mean_aggregation_flag = fused_message_mean_aggregation_flag

        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.node_dict = mydglgraph.get_ntype_dict()
        # self.edge_dict = mydglgraph.get_etype_dict()
        self.num_ntypes = mydglgraph.get_num_ntypes()  # len(self.node_dict)
        self.num_relations = mydglgraph.get_num_rels()  # len(self.edge_dict)

        self.multiply_among_weights_first_flag = multiply_among_weights_first_flag

        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.src_node_type_per_canonical_edge_type = (
            src_node_type_per_canonical_edge_type
        )
        self.dst_node_type_per_canonical_edge_type = (
            dst_node_type_per_canonical_edge_type
        )

        # self.k_linears = nn.ModuleList()
        # self.q_linears = nn.ModuleList()
        # self.v_linears = nn.ModuleList()
        # self.a_linears = nn.ModuleList()
        if self.multiply_among_weights_first_flag:
            # this is a varient of the original weights where first the view is changed from (self.num_ntypes, 1, in_dim, out_dim) to (self.num_ntypes, in_dim, self.n_heads, self.d_k) and then a transposition is applied
            self.k_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, self.n_heads, in_dim, self.d_k)
            )
            self.q_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, self.n_heads, in_dim, self.d_k)
            )
            self.v_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, self.n_heads, in_dim, self.d_k)
            )
        else:
            self.k_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, 1, in_dim, out_dim)
            )
            self.q_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, 1, in_dim, out_dim)
            )
            self.v_linears = nn.Parameter(
                torch.Tensor(self.num_ntypes, 1, in_dim, out_dim)
            )

        self.a_linears = nn.Parameter(
            torch.Tensor(self.num_ntypes, 1, out_dim, out_dim)
        )
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_ntypes):
            # self.k_linears.append(nn.Linear(in_dim, out_dim))
            # self.q_linears.append(nn.Linear(in_dim, out_dim))
            # self.v_linears.append(nn.Linear(in_dim, out_dim))
            # self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_ntypes, 1, 1, 1))
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
        nn.init.xavier_uniform_(self.k_linears)
        nn.init.xavier_uniform_(self.q_linears)
        nn.init.xavier_uniform_(self.v_linears)
        nn.init.xavier_uniform_(self.a_linears)
        # for module in self.k_linears:
        #    module.reset_parameters()
        # for module in self.q_linears:
        #    module.reset_parameters()
        # for module in self.v_linears:
        #    module.reset_parameters()
        # for module in self.a_linears:
        #     module.reset_parameters()

    # @torch.jit.script_method
    def forward(self, G, h):
        # G is MyDGLGraph, h is node features with shape (num_nodes, in_dim).
        # We assume h is made up of one continuous memory region, where each node type occupies one continuous subregion.

        if self.multiply_among_weights_first_flag:
            k_per_canonical_etype = torch.index_select(
                self.k_linears, 0, self.src_node_type_per_canonical_edge_type
            )
            q_per_canonical_etype = torch.index_select(
                self.q_linears, 0, self.dst_node_type_per_canonical_edge_type
            )
            v_per_canonical_etype = torch.index_select(
                self.v_linears, 0, self.dst_node_type_per_canonical_edge_type
            )
            # fixme: the view api does not do transpose
            k_relation_att_q_product = torch.bmm(
                k_per_canonical_etype.view(-1, self.in_dim, self.d_k),
                self.relation_att.view(-1, self.d_k, self.d_k),
            ).view(-1, self.n_heads, self.in_dim, self.d_k)
            k_relation_att_q_product = torch.bmm(
                k_relation_att_q_product.view(-1, self.in_dim, self.d_k),
                q_per_canonical_etype.view(-1, self.in_dim, self.d_k).transpose(1, 2),
            ).view(-1, self.n_heads, self.in_dim, self.in_dim)
            relation_msg_v_product = torch.bmm(
                self.relation_msg.view(-1, self.d_k, self.d_k),
                v_per_canonical_etype.view(-1, self.d_k, self.d_k),
            ).view(-1, self.n_heads, self.in_dim, self.d_k)

            relation_att_weight = k_relation_att_q_product.contiguous()
            relation_msg_weight = relation_msg_v_product.contiguous()
            k = h.unsqueeze(1).repeat(1, self.n_heads, 1).contiguous()
            q = k
            v = k
        else:
            relation_att_weight = self.relation_att
            relation_msg_weight = self.relation_msg
            k = B.rgnn_relational_matmul_no_scatter_gather_list(
                G.get_original_node_type_offsets(), self.k_linears, h
            ).view(-1, self.n_heads, self.d_k)
            q = B.rgnn_relational_matmul_no_scatter_gather_list(
                G.get_original_node_type_offsets(), self.q_linears, h
            ).view(-1, self.n_heads, self.d_k)
            v = B.rgnn_relational_matmul_no_scatter_gather_list(
                G.get_original_node_type_offsets(), self.v_linears, h
            ).view(-1, self.n_heads, self.d_k)

        if self.hgt_fused_attn_score_flag:
            attn_score = B.hgt_full_graph_hetero_attention_ops_coo(
                G, relation_att_weight, k, q
            )  # shape (num_edges, n_heads)
        else:
            if self.compact_as_of_node_flag:
                separate_coo_original_dict = G.get_separate_coo_original()
                separate_unique_node_indices_dict = G.get_separate_unique_node_indices()
                # TODO: implement single-sided matmul for compact_as_of_node_flag
                attn_weight_dst_product_compact = (
                    B.rgnn_relational_matmul_compact_as_of_node_single_ended(
                        separate_unique_node_indices_dict["rel_ptr"],
                        separate_unique_node_indices_dict["node_idx"],
                        separate_coo_original_dict["rel_ptr"],
                        separate_coo_original_dict["col_idx"],
                        separate_coo_original_dict["eids"],
                        relation_att_weight,
                        q,
                        False,
                    )
                )
                attn_score = B.rgnn_inner_product_node_compact_and_node(
                    separate_unique_node_indices_dict["rel_ptr"],
                    separate_unique_node_indices_dict["node_idx"],
                    separate_coo_original_dict["rel_ptr"],
                    separate_coo_original_dict["eids"],
                    separate_coo_original_dict["row_idx"],
                    separate_coo_original_dict["col_idx"],
                    attn_weight_dst_product_compact,
                    k,
                )
            else:
                separate_coo_original_dict = G.get_separate_coo_original()
                attn_weight_dst_product_per_edge = B.rgnn_relational_matmul(
                    separate_coo_original_dict["rel_ptr"],
                    separate_coo_original_dict["col_idx"],
                    separate_coo_original_dict["eids"],
                    relation_att_weight,
                    q,
                    False,
                )
                attn_score = B.rgnn_inner_product_edge_and_node(
                    separate_coo_original_dict["eids"],
                    separate_coo_original_dict["row_idx"],
                    separate_coo_original_dict["col_idx"],
                    attn_weight_dst_product_per_edge,
                    k,
                )

        # attn_score = B.hgt_full_graph_edge_softmax_ops_csr(
        #    G, attn_score, mu=(self.relation_pri / self.sqrt_dk)
        # )
        # NB: the scaling is: attn_score = relation_pri / self.sqrt_dk * attn_score

        if self.fused_message_mean_aggregation_flag:
            separate_coo_original_dict = G.get_separate_coo_original()
            new_h = B.hgt_full_graph_message_calc_edge_softmax_and_message_mean_aggregation_csr(
                separate_coo_original_dict["rel_ptr"],
                separate_coo_original_dict["row_idx"],
                separate_coo_original_dict["col_idx"],
                separate_coo_original_dict["eids"],
                relation_msg_weight,
                v,
                G,
                (self.relation_pri / self.sqrt_dk),
                attn_score,
            )
        else:
            separate_coo_original_dict = G.get_separate_coo_original()
            message_per_edge = B.rgnn_relational_matmul(
                separate_coo_original_dict["rel_ptr"],
                separate_coo_original_dict["row_idx"],
                separate_coo_original_dict["eids"],
                relation_msg_weight,
                v,
                False,
            )  # shape (num_edges, n_heads, d_k)
            new_h = B.hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr(
                G,
                message_per_edge,
                attn_score,
                (self.relation_pri / self.sqrt_dk),
                # g["separate"]["unique_node_idx"]["rel_ptr"],
                # g["separate"]["unique_node_idx"]["node_idx"],
            )  # shape (num_nodes, n_heads, d_k)

        new_h = B.rgnn_relational_matmul_no_scatter_gather_list(
            G.get_original_node_type_offsets(),
            (torch.sigmoid(self.skip) * self.a_linears),
            new_h,
        )

        return new_h


class HET_HGT_DGLHetero(nn.Module):
    @utils_lite.warn_default_arguments
    def __init__(
        self,
        mydglgraph,
        in_dim,
        out_dim,
        src_node_type_per_canonical_edge_type: torch.Tensor,
        dst_node_type_per_canonical_edge_type: torch.Tensor,
        n_heads=1,
        dropout=0.2,
        multiply_among_weights_first_flag=False,
        hgt_fused_attn_score_flag=False,
        compact_as_of_node_flag=False,
        fused_message_mean_aggregation_flag=False,
    ):  # ,h_dim
        super(HET_HGT_DGLHetero, self).__init__()
        self.mydglgraph = mydglgraph
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        # self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer0 = HET_HGTLayerHetero(
            in_dim,
            out_dim,
            mydglgraph,
            src_node_type_per_canonical_edge_type,
            dst_node_type_per_canonical_edge_type,
            n_heads=n_heads,
            dropout=dropout,
            multiply_among_weights_first_flag=multiply_among_weights_first_flag,
            hgt_fused_attn_score_flag=hgt_fused_attn_score_flag,
            compact_as_of_node_flag=compact_as_of_node_flag,
            fused_message_mean_aggregation_flag=fused_message_mean_aggregation_flag,
        )
        # self.layer1 = HET_HGTLayerHetero(
        #    h_dim, out_dim, mydglgraph, n_heads=n_heads, dropout=dropout
        # )

    def reset_parameters(self):
        self.layer0.reset_parameters()
        # self.layer1.reset_parameters()

    def const_to(self, device):
        self.layer0.src_node_type_per_canonical_edge_type = (
            self.layer0.src_node_type_per_canonical_edge_type.to(device)
        )
        self.layer0.dst_node_type_per_canonical_edge_type = (
            self.layer0.dst_node_type_per_canonical_edge_type.to(device)
        )

    def forward(self, h):
        h = self.layer0(self.mydglgraph, h)
        # h = self.layer1(G, h)
        return h
