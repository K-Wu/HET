#!/usr/bin/env python3
# from graphiler/examples/HGT/HGT_DGL.py
import math

import torch
import torch.nn as nn

# import dgl.function as fn
# from dgl.nn.functional import edge_softmax
from .. import backend as B
from .. import utils


class HET_HGTLayerHetero(nn.Module):
    @utils.warn_default_arguments
    def __init__(
        self,
        in_dim,
        out_dim,
        mydglgraph,
        n_heads=1,
        dropout=0.2,
        use_norm=False,
    ):
        super(HET_HGTLayerHetero, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = mydglgraph.get_ntype_dict()
        self.edge_dict = mydglgraph.get_etype_dict()
        self.num_types = len(self.node_dict)
        self.num_relations = len(self.edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        # G is MyDGLGraph, h is node features with shape (num_nodes, in_dim).
        # We assume h is made up of one continuous memory region, where each node type occupies one continuous subregion.
        # TODO: add node type offset to G.

        node_dict, edge_dict = self.node_dict, self.edge_dict

        src_nodetypes = set()
        dest_nodetypes = set()
        for srctype, etype, dsttype in G.canonical_etypes:
            assert (
                srctype in G.ntypes
            ), "srctype not in G.ntypes. Maybe ambiguity in node types?"
            assert (
                dsttype in G.ntypes
            ), "dsttype not in G.ntypes. Maybe ambiguity in node types?"
            src_nodetypes.add(srctype)
            dest_nodetypes.add(dsttype)

        k = torch.zeros((G.number_of_nodes(), self.n_heads, self.d_k), device=h.device)
        q = torch.zeros((G.number_of_nodes(), self.n_heads, self.d_k), device=h.device)
        v = torch.zeros((G.number_of_nodes(), self.n_heads, self.d_k), device=h.device)

        for srctype in src_nodetypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            k[
                G["original"]["node_type_offsets"][srctype] : G["original"][
                    "node_type_offsets"
                ][srctype + 1]
            ] = k_linear(
                h[
                    G["original"]["node_type_offsets"][srctype] : G["original"][
                        "node_type_offsets"
                    ][srctype + 1]
                ]
            ).view(
                -1, self.n_heads, self.d_k
            )
            v[
                G["original"]["node_type_offsets"][srctype] : G["original"][
                    "node_type_offsets"
                ][srctype + 1]
            ] = v_linear(
                h[
                    G["original"]["node_type_offsets"][srctype] : G["original"][
                        "node_type_offsets"
                    ][srctype + 1]
                ]
            ).view(
                -1, self.n_heads, self.d_k
            )

        for dsttype in dest_nodetypes:
            q_linear = self.q_linears[node_dict[dsttype]]
            q[
                G["original"]["node_type_offsets"][dsttype] : G["original"][
                    "node_type_offsets"
                ][dsttype + 1]
            ] = q_linear(
                h[
                    G["original"]["node_type_offsets"][dsttype] : G["original"][
                        "node_type_offsets"
                    ][dsttype + 1]
                ]
            ).view(
                -1, self.n_heads, self.d_k
            )

        message_per_edge = B.rgnn_relational_matmul(
            G["separate"]["coo"]["original"]["rel_ptr"],
            G["separate"]["coo"]["original"]["row_idx"],
            G["separate"]["coo"]["original"]["eids"],
            self.relation_msg,
            v,
        )  # shape (num_edges, n_heads, d_k)

        if self.hgt_fused_attn_score_flag:
            attn_score = B.hgt_full_graph_hetero_attention_ops_csr(
                G, self.relation_att, k, q
            )  # shape (num_edges, n_heads)
        else:
            if self.compact_as_of_node_flag:
                # TODO: implement single-sided matmul for compact_as_of_node_flag
                attn_weight_dst_product_compact = (
                    B.rgnn_relational_matmul_compact_as_of_node_single_ended(
                        G["separate"]["unique_node_idx"]["rel_ptr"],
                        G["separate"]["unique_node_idx"]["node_idx"],
                        G["separate"]["coo"]["original"]["rel_ptr"],
                        G["separate"]["coo"]["original"]["col_idx"],
                        self.relation_att,
                        q,
                    )
                )
                attn_score = B.rgnn_inner_product_node_compact_and_node(
                    G["separate"]["unique_node_idx"]["rel_ptr"],
                    G["separate"]["unique_node_idx"]["node_idx"],
                    G["separate"]["coo"]["original"]["rel_ptr"],
                    G["separate"]["coo"]["original"]["eids"],
                    G["separate"]["coo"]["original"]["col_idx"],
                    attn_weight_dst_product_compact,
                    k,
                )
            else:
                attn_weight_dst_product_per_edge = B.rgnn_relational_matmul(
                    G["separate"]["coo"]["original"]["rel_ptr"],
                    G["separate"]["coo"]["original"]["col_idx"],
                    G["separate"]["coo"]["original"]["eids"],
                    self.relation_att,
                    q,
                )
                attn_score = B.rgnn_inner_product_edge_and_node(
                    G["separate"]["coo"]["original"]["eids"],
                    G["separate"]["coo"]["original"]["col_idx"],
                    attn_weight_dst_product_per_edge,
                    k,
                )

        # attn_score = B.hgt_full_graph_edge_softmax_ops_csr(
        #    G, attn_score, mu=(self.relation_pri / self.sqrt_dk)
        # )
        # NB: the scaling is: attn_score = relation_pri / self.sqrt_dk * attn_score

        new_h = B.hgt_full_graph_edge_softmax_and_message_mean_aggregation_csr(
            G,
            message_per_edge,
            attn_score,
            (self.relation_pri / self.sqrt_dk),
            # g["separate"]["unique_node_idx"]["rel_ptr"],
            # g["separate"]["unique_node_idx"]["node_idx"],
        )  # shape (num_nodes, n_heads, d_k)

        new_h_normed = torch.zeros((G.number_of_nodes(), self.out_dim), device=h.device)
        for dsttype in dest_nodetypes:
            """
            Step 3: Target-specific Aggregation
            x = norm( W[node_type] * gelu( Agg(x) ) + x )
            """
            n_id = node_dict[dsttype]
            alpha = torch.sigmoid(self.skip[n_id])

            trans_out = self.drop(
                self.a_linears[n_id](
                    new_h[
                        G["original"]["node_type_offsets"][dsttype] : G["original"][
                            "node_type_offsets"
                        ][dsttype + 1]
                    ]
                )
            )
            trans_out = trans_out * alpha  # + h[ntype] * (1-alpha) ?
            if self.use_norm:
                new_h_normed[
                    G["original"]["node_type_offsets"][dsttype] : G["original"][
                        "node_type_offsets"
                    ][dsttype + 1]
                ] = self.norms[n_id](trans_out)
            else:
                new_h_normed[
                    G["original"]["node_type_offsets"][dsttype] : G["original"][
                        "node_type_offsets"
                    ][dsttype + 1]
                ] = trans_out
        return new_h_normed


class HET_HGT_DGLHetero(nn.Module):
    @utils.warn_default_arguments
    def __init__(self, mydglgraph, in_dim, out_dim, n_heads=1, dropout=0.2):  # ,h_dim
        super(HET_HGT_DGLHetero, self).__init__()
        self.mydglgraph = mydglgraph
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        # self.h_dim = h_dim
        self.out_dim = out_dim
        self.layer0 = HET_HGTLayerHetero(
            in_dim, out_dim, mydglgraph, n_heads=n_heads, dropout=dropout
        )
        # self.layer1 = HET_HGTLayerHetero(
        #    h_dim, out_dim, mydglgraph, n_heads=n_heads, dropout=dropout
        # )

    def forward(self, G, h):
        h = self.layer0(G, h)
        # h = self.layer1(G, h)
        return h
