#!/usr/bin/env python3
# import itertools
from typing import Union
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl

# import dgl.nn as dglnn
from .. import backend as B
from .. import utils


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
    num_rels: int
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

    @utils.warn_default_arguments
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        n_heads,
        *,
        bias: bool = True,
        activation=None,
        self_loop: bool = False,
        compact_as_of_node_flag: bool = False,
        multiply_among_weights_first_flag: bool = False,
        dropout=0.5,
        leaky_relu_slope=0.2,
    ):
        super(HET_RelationalAttLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.multiply_among_weights_first_flag = multiply_among_weights_first_flag
        self.leaky_relu_slope = leaky_relu_slope

        assert (
            num_rels > 1
        ), "dummy proof assertion num_rels should be larger than 1 normally when calling RGAT_HET"
        # NB: RGAT model definition
        assert out_feat % n_heads == 0, "out_feat must be a multiple of n_heads"

        self.conv_weights = nn.Parameter(
            th.Tensor(num_rels, n_heads, in_feat, out_feat // n_heads)
        )
        self.attn_l = nn.Parameter(th.Tensor(num_rels, n_heads, out_feat // n_heads))
        self.attn_r = nn.Parameter(th.Tensor(num_rels, n_heads, out_feat // n_heads))
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
    def forward(self, g, inputs: th.Tensor, myblock: Union[None, list] = None):
        """Forward computation

        Parameters
        ----------
        g : MyDGLGraph or ScriptedMyDGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """

        if myblock is not None:
            raise NotImplementedError("Block is not supported by us yet")
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # NB: this line originally calls DGL R(GAT) impl and is now replaced with our own logic
        # hs = B.rgat_layer_csr(g, self.conv_weights, inputs)
        if self.compact_as_of_node_flag:
            # feat_src_compact = B.rgnn_relational_matmul_compact_as_of_node(
            #     g, self.conv_weight, inputs
            # )
            # feat_dst_compact = B.rgnn_relational_matmul_compact_as_of_node(
            #     g["separate"]["unique_node_idx"]["rel_ptr"],g["separate"]["unique_node_idx"]["node_idx"], self.conv_weight, inputs
            # )
            if self.multiply_among_weights_first_flag:
                product_of_conv_weights_attn_r = (
                    self.conv_weights * self.attn_r.unsqueeze(2)
                )
                product_of_conv_weights_attn_l = (
                    self.conv_weights * self.attn_l.unsqueeze(2)
                )
                separate_unique_node_idx = g.get_separate_unique_node_indices()
                feat_compact = B.rgnn_relational_matmul_compact_as_of_node(
                    separate_unique_node_idx["rel_ptr"],
                    separate_unique_node_idx["node_idx"],
                    self.conv_weights,
                    inputs,
                    True,
                )
                el_compact = B.rgnn_relational_matmul_compact_as_of_node(
                    separate_unique_node_idx["rel_ptr"],
                    separate_unique_node_idx["node_idx"],
                    product_of_conv_weights_attn_l,
                    inputs,
                    True,
                )
                er_compact = B.rgnn_relational_matmul_compact_as_of_node(
                    separate_unique_node_idx["rel_ptr"],
                    separate_unique_node_idx["node_idx"],
                    product_of_conv_weights_attn_r,
                    inputs,
                    True,
                )
                el_compact = el_compact.sum(dim=-1)
                er_compact = er_compact.sum(dim=-1)
            else:
                separate_unique_node_idx = g.get_separate_unique_node_indices()
                # TODO: separate feat_compact_src and feat_compact_dst
                feat_compact = B.rgnn_relational_matmul_compact_as_of_node(
                    separate_unique_node_idx["rel_ptr"],
                    separate_unique_node_idx["node_idx"],
                    self.conv_weights,
                    inputs,
                    True,
                )
                # FIXME: the following two lines should be implemented with relational_inner_product_compact_and_weight
                el_compact = (feat_compact * self.attn_l).sum(dim=-1).unsqueeze(-1)
                er_compact = (feat_compact * self.attn_r).sum(dim=-1).unsqueeze(-1)
            h = B.relational_fused_gat_compact_as_of_node(
                g, feat_compact, el_compact, er_compact, self.leaky_relu_slope
            )

        else:
            separate_coo_original_dict = g.get_separate_coo_original()
            # print(th.argmin(g["separate"]["coo"]["original"]["eids"])) # 256546 rel_ptr [183, 184)
            feat_src_per_edge = B.rgnn_relational_matmul(
                separate_coo_original_dict["rel_ptr"],
                separate_coo_original_dict["row_idx"],
                separate_coo_original_dict["eids"],
                # g["separate"]["unique_node_idx"]["rel_ptr"],
                # g["separate"]["unique_node_idx"]["node_idx"],
                self.conv_weights,
                inputs,
                True,
            )
            el = B.rgnn_relational_matmul(
                separate_coo_original_dict["rel_ptr"],
                separate_coo_original_dict["eids"],
                separate_coo_original_dict["eids"],
                self.attn_l.unsqueeze(-1),
                feat_src_per_edge,
                False,
            )
            # el = (feat_src_per_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)

            if self.multiply_among_weights_first_flag:
                # attn_r_reshaped = th.reshape(
                #    self.attn_r, (-1, self.n_heads, self.out_feat // self.n_heads, 1)
                # )
                # conv_weights_reshaped = th.reshape(
                #    self.conv_weights,
                #    (-1, self.n_heads, self.in_feat, self.out_feat // self.n_heads),
                # )
                #
                # product_of_conv_weights_attn_r = th.bmm(
                #    conv_weights_reshaped, attn_r_reshaped
                # )
                # product_of_conv_weights_attn_r = th.reshape(
                #    product_of_conv_weights_attn_r, (-1, self.n_heads, self.in_feat)
                # )
                separate_coo_original_dict = g.get_separate_coo_original()
                product_of_conv_weights_attn_r = (
                    self.conv_weights * self.attn_r.unsqueeze(2)
                )
                er = B.rgnn_relational_matmul(
                    separate_coo_original_dict["rel_ptr"],
                    separate_coo_original_dict["eids"],
                    separate_coo_original_dict["eids"],
                    product_of_conv_weights_attn_r,
                    inputs,
                    True,
                )
            else:
                separate_coo_original_dict = g.get_separate_coo_original()
                feat_dst_per_edge = B.rgnn_relational_matmul(
                    separate_coo_original_dict["rel_ptr"],
                    separate_coo_original_dict["col_idx"],
                    separate_coo_original_dict["eids"],
                    # g["separate"]["unique_node_idx"]["rel_ptr"],
                    # g["separate"]["unique_node_idx"]["node_idx"],
                    self.conv_weights,
                    inputs,
                    True,
                )
                er = B.rgnn_relational_matmul(
                    separate_coo_original_dict["rel_ptr"],
                    separate_coo_original_dict["eids"],
                    separate_coo_original_dict["eids"],
                    self.attn_r.unsqueeze(-1),
                    feat_dst_per_edge,
                    False,
                )
                # er = (feat_dst_per_edge * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # print("el", el.shape)
            # print("er", er.shape)
            # print("max eids", g["separate"]["coo"]["original"]["eids"].max())
            # print("max eids", g["original"]["eids"].max())

            h = B.relational_fused_gat_csr(
                g, feat_src_per_edge, el, er, self.leaky_relu_slope
            )

        # hs = self.conv(g, inputs_src)

        # NB: let's leverage the built-in bias, activation and dropout here and only focus on SpMM/SDDMM in our kernel implementation.
        # NB: GATConv class also provides bias, activation and dropout but we can ignore them for now.

        h = h.view(-1, self.out_feat)
        if self.self_loop:
            # print(inputs_dst.shape)
            h = h + th.matmul(inputs_dst, self.loop_weight)
        if self.bias:
            h = h + self.h_bias
        if self.activation:
            h = self.activation(h)
        return self.dropout(h)


class HET_RelationalGATEncoder(nn.Module):
    # corresponding to EntityClassify in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention encoder

    Parameters
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

    @utils.warn_default_arguments
    def __init__(
        self,
        num_etypes,
        h_dim,
        out_dim,
        n_heads,
        num_hidden_layers=1,
        dropout=0.5,
        use_self_loop: bool = True,
        last_layer_act: bool = False,
        compact_as_of_node_flag: bool = False,
        multiply_among_weights_first_flag: bool = False,
    ):
        super(HET_RelationalGATEncoder, self).__init__()
        self.n_heads = n_heads
        self.num_etypes = num_etypes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.last_layer_act = last_layer_act
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.multiply_among_weights_first_flag = multiply_among_weights_first_flag
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
                    self.num_etypes,
                    self.n_heads,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                    compact_as_of_node_flag=self.compact_as_of_node_flag,
                    multiply_among_weights_first_flag=self.multiply_among_weights_first_flag,
                )
            )
        # h2o
        self.layers.append(
            HET_RelationalAttLayer(
                self.h_dim,
                self.out_dim,
                self.num_etypes,
                1,  # overwrting the n_head setting as the classification should be output in this stage
                activation=F.relu if self.last_layer_act else None,
                self_loop=self.use_self_loop,
                compact_as_of_node_flag=self.compact_as_of_node_flag,
                multiply_among_weights_first_flag=self.multiply_among_weights_first_flag,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        g,
        h: Union[th.Tensor, None] = None,
        blocks: Union[None, list] = None,
    ):
        """Forward computation

        Parameters
        ----------
        g: mydglgraph
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        """
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h
