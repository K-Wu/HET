#!/usr/bin/env python3
# import itertools
import torch as th
from torch import nn
import torch.nn.functional as F

# import dgl.nn as dglnn
from .. import backend as B
from .models_dgl import RGAT_main_procedure, RGAT_parse_args


class HET_RelationalAttLayer(nn.Module):
    # corresponding to RelGraphConvLayer in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention layer.

    For inner relation message aggregation we use multi-head attention network.
    For cross relation message we just use average

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    n_heads : int
        Number of attention heads
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        n_heads,
        *,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
    ):
        super(HET_RelationalAttLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        # NB: RGAT model definition
        assert out_feat % n_heads == 0, "out_feat must be a multiple of n_heads"

        self.conv_weights = nn.Parameter(
            th.Tensor(len(rel_names), n_heads, in_feat, out_feat // n_heads)
        )
        self.attn_l = nn.Parameter(
            th.Tensor(len(rel_names), n_heads, out_feat // n_heads)
        )
        self.attn_r = nn.Parameter(
            th.Tensor(len(rel_names), n_heads, out_feat // n_heads)
        )
        # self.conv = dglnn.HeteroGraphConv(
        #     {
        #         rel: dglnn.GATConv(in_feat, out_feat // n_heads, n_heads, bias=False)
        #         for rel in rel_names
        #     }
        # )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))

        # self.reset_parameters()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        if self.bias:
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        nn.init.xavier_uniform_(self.conv_weights, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.attn_l, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.attn_r, gain=nn.init.calculate_gain("relu"))

        # for module in self.conv.mods.values():
        #     module.reset_parameters()

    # pylint: disable=invalid-name
    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # NB: this line originally calls DGL R(GAT) impl and is now replaced with our own logic
        # hs = B.rgat_layer_csr(g, self.conv_weights, input)
        if self.compact_as_of_node_flag:
            # feat_src_compact = B.rgat_relational_matmul_compact_as_of_node(
            #     g, self.conv_weight, input
            # )
            # feat_dst_compact = B.rgat_relational_matmul_compact_as_of_node(
            #     g["separate"]["unique_srcs_and_dests"]["rel_ptr"],g["separate"]["unique_srcs_and_dests"]["unique_node_idxes"], self.conv_weight, input
            # )
            feat_compact = B.rgat_relational_matmul_compact_as_of_node(
                g["separate"]["unique_srcs_and_dests"]["rel_ptr"],
                g["separate"]["unique_srcs_and_dests"]["unique_node_idxes"],
                self.conv_weight,
                input,
            )
            el_compact = (feat_compact * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er_compact = (feat_compact * self.attn_r).sum(dim=-1).unsqueeze(-1)
            h = B.relational_fused_gat_compact_as_of_node(
                g, feat_compact, el_compact, er_compact, self.negative_slope
            )

        else:
            feat_src_per_edge = B.rgat_relational_matmul(
                g["separate"]["coo"]["original"]["rel_ptr"],
                g["separate"]["coo"]["original"]["row_idx"],
                g["separate"]["coo"]["original"]["eids"],
                self.conv_weights,
                input,
            )
            feat_dst_per_edge = B.rgat_relational_matmul(
                g["separate"]["coo"]["original"]["rel_ptr"],
                g["separate"]["coo"]["original"]["col_idx"],
                g["separate"]["coo"]["original"]["eids"],
                self.conv_weights,
                input,
            )
            el = (feat_src_per_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst_per_edge * self.attn_r).sum(dim=-1).unsqueeze(-1)
            h = B.relational_fused_gat_csr(
                g, feat_src_per_edge, el, er, self.negative_slope
            )

        # hs = self.conv(g, inputs_src)

        # NB: let's leverage the built-in bias, activation and dropout here and only focus on SpMM/SDDMM in our kernel implementation.
        # NB: GATConv class also provides bias, activation and dropout but we can ignore them for now.

        h = h.view(-1, self.out_feat)
        if self.self_loop:
            h = h + th.matmul(h, self.loop_weight)
        if self.bias:
            h = h + self.h_bias
        if self.activation:
            h = self.activation(h)
        return self.dropout(h)


class HET_RelationalGATEncoder(nn.Module):
    # corresponding to EntityClassify in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention encoder

    Parameters
    g : DGLHeteroGraph
        Input graph.
    h_dim: int
        Hidden dimension size
    out_dim: int
        Output dimension size
    n_heads: int
        Number of heads
    num_hidden_layers: int
        Num hidden layers
    dropout: float
        Dropout
    use_self_loop: bool
        Self loop
    last_layer_act: bool
        Whether add activation at the last layer
    """

    def __init__(
        self,
        g,
        h_dim,
        out_dim,
        n_heads,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=True,
        last_layer_act=False,
    ):
        super(HET_RelationalGATEncoder, self).__init__()
        self.n_heads = n_heads
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.last_layer_act = last_layer_act
        self.init_encoder()

    def init_encoder(self):
        """Initialize RelationalGATEncoder encoder"""
        self.layers = nn.ModuleList()
        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                HET_RelationalAttLayer(
                    self.h_dim,
                    self.h_dim,
                    self.g.etypes,
                    self.n_heads,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        # h2o
        self.layers.append(
            HET_RelationalAttLayer(
                self.h_dim,
                self.out_dim,
                self.g.etypes,
                1,  # overwrting the n_head setting as the classification should be output in this stage
                activation=F.relu if self.last_layer_act else None,
                self_loop=self.use_self_loop,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, h=None, blocks=None):
        """Forward computation

        Parameters
        ----------
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        """
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h


class HET_RelationalGATEncoderSingleLayer(HET_RelationalGATEncoder):
    def __init__(
        self,
        g,
        h_dim,
        out_dim,
        n_heads,
        dropout=0,
        use_self_loop=True,
        last_layer_act=False,
    ):
        super(HET_RelationalGATEncoderSingleLayer, self).__init__(
            g,
            h_dim,
            out_dim,
            n_heads,
            num_hidden_layers=0,
            dropout=dropout,
            use_self_loop=use_self_loop,
            last_layer_act=last_layer_act,
        )


if __name__ == "__main__":
    args = RGAT_parse_args()
    RGAT_main_procedure(args, dgl_model_flag=False)
