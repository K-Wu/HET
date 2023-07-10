#!/usr/bin/env python3
# import itertools
from typing import Union
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl

# import nvtx

# import dgl.nn as dglnn
from .. import backend as B
from .. import utils_lite


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
    num_heads : int
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

    @utils_lite.warn_default_arguments
    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        num_heads,
        *,
        bias: bool = True,
        activation=None,
        self_loop: bool = False,
        compact_as_of_node_flag: bool = False,
        compact_direct_indexing_flag: bool = False,
        multiply_among_weights_first_flag: bool = False,
        gat_edge_parallel_flag: bool = False,
        dropout=0.5,
        leaky_relu_slope=0.2,
    ):
        super(HET_RelationalAttLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.compact_direct_indexing_flag = compact_direct_indexing_flag
        self.multiply_among_weights_first_flag = multiply_among_weights_first_flag
        self.gat_edge_parallel_flag = gat_edge_parallel_flag
        self.leaky_relu_slope = leaky_relu_slope

        assert (
            num_rels > 1
        ), "dummy proof assertion num_rels should be larger than 1 normally when calling RGAT_HET"
        # NB: RGAT model definition
        assert out_feat % num_heads == 0, "out_feat must be a multiple of num_heads"

        self.conv_weights = nn.Parameter(
            th.Tensor(num_rels, num_heads, in_feat, out_feat // num_heads)
        )
        self.attn_l = nn.Parameter(
            th.Tensor(num_rels, num_heads, out_feat // num_heads)
        )
        self.attn_r = nn.Parameter(
            th.Tensor(num_rels, num_heads, out_feat // num_heads)
        )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))

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
        # with nvtx.annotate("forward", color="purple"):
        if myblock is not None:
            raise NotImplementedError("Block is not supported by us yet")
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        # NB: this line originally calls DGL R(GAT) impl and is now replaced with our own logic
        if self.compact_as_of_node_flag:
            matmul_compact_as_of_node_kind = (
                1  # 2 if self.compact_direct_indexing_flag else 1
            )
            if self.compact_direct_indexing_flag:
                print(
                    "Warning: matmul currently ignore compact direct indexing flag as it is unclear how to benefit from it"
                )
            separate_unique_node_indices_single_sided = (
                g.get_separate_unique_node_indices_single_sided()
            )
            # TODO: check if it is okay to pass single sided unique node indices to compact relational_matmul
            args_tensor_dict_row = {
                "unique_srcs_and_dests_rel_ptrs": separate_unique_node_indices_single_sided[
                    "rel_ptrs_row"
                ],
                "unique_srcs_and_dests_node_indices": separate_unique_node_indices_single_sided[
                    "node_indices_row"
                ],
            }
            args_tensor_dict_col = {
                "unique_srcs_and_dests_rel_ptrs": separate_unique_node_indices_single_sided[
                    "rel_ptrs_col"
                ],
                "unique_srcs_and_dests_node_indices": separate_unique_node_indices_single_sided[
                    "node_indices_col"
                ],
            }
            if self.multiply_among_weights_first_flag:
                product_of_conv_weights_attn_r = th.bmm(
                    self.conv_weights.view(
                        -1, self.in_feat, self.out_feat // self.num_heads
                    ),
                    self.attn_r.view(-1, self.out_feat // self.num_heads, 1),
                ).view(-1, self.num_heads, self.in_feat, 1)
                product_of_conv_weights_attn_l = th.bmm(
                    self.conv_weights.view(
                        -1, self.in_feat, self.out_feat // self.num_heads
                    ),
                    self.attn_l.view(-1, self.out_feat // self.num_heads, 1),
                ).view(-1, self.num_heads, self.in_feat, 1)
                feat_compact = B.rgnn_relational_matmul(
                    args_tensor_dict_row,
                    self.conv_weights,
                    inputs,
                    True,  # fixme: check if this is correct
                    matmul_compact_as_of_node_kind,  # CompactAsOfNodeKind::Enabled or Direct Index
                )  # NB: use single side instead without need to modify kernel
                el_compact = B.rgnn_relational_matmul(
                    args_tensor_dict_row,
                    product_of_conv_weights_attn_l,
                    inputs,
                    True,
                    matmul_compact_as_of_node_kind,  # CompactAsOfNodeKind::Enabled or Direct Index
                )  # NB: use single side instead without need to modify kernel
                er_compact = B.rgnn_relational_matmul(
                    args_tensor_dict_col,
                    product_of_conv_weights_attn_r,
                    inputs,
                    True,
                    matmul_compact_as_of_node_kind,  # CompactAsOfNodeKind::Enabled or Direct Index
                )  # NB: use single side instead without need to modify kernel
            else:
                separate_unique_node_indices_single_sided = (
                    g.get_separate_unique_node_indices_single_sided()
                )
                # NB: no need to distinguish feat_compact_src and feat_compact_dst because in our case all datasets are added with inverse edges
                feat_compact = B.rgnn_relational_matmul(
                    args_tensor_dict_row,
                    self.conv_weights,
                    inputs,
                    True,
                    matmul_compact_as_of_node_kind,  # CompactAsOfNodeKind::Enabled or Direct Index
                )  # NB: use single side instead without need to modify kernel
                feat_compact_dst = B.rgnn_relational_matmul(
                    args_tensor_dict_col,
                    self.conv_weights,
                    inputs,
                    True,
                    matmul_compact_as_of_node_kind,  # CompactAsOfNodeKind::Enabled or Direct Index
                )  # NB: use single side instead without need to modify kernel
                # notice that rgnn_inner_product_right_node is not applicable here because weight is not right-hand-side data. So it should be something like relational_inner_product_compact_and_weight. We use the GEMM for convenience as an ad-hoc solution

                # TODO: for performance, the following two lines may as well be implemented with relational_inner_product_compact_and_weight rather than GEMM

                # Originally, the fw and bck kernels assert head == 1 because they were originally implemented for HGT.
                # We generalize them so that they can work in this case even if num_head > 1
                # print(feat_compact.shape, self.attn_l.unsqueeze(-1).shape)
                el_compact = B.rgnn_relational_matmul_no_scatter_gather_list(
                    separate_unique_node_indices_single_sided["rel_ptrs_row"],
                    self.attn_l.unsqueeze(-1),
                    feat_compact,
                )  # NB: use single side instead without need to modify kernel
                er_compact = B.rgnn_relational_matmul_no_scatter_gather_list(
                    separate_unique_node_indices_single_sided["rel_ptrs_col"],
                    self.attn_r.unsqueeze(-1),
                    feat_compact_dst,
                )  # NB: use single side instead without need to modify kernel

            if self.gat_edge_parallel_flag:  # NB: use a flag to switch this
                h = B.relational_fused_gat_compact_as_of_node_separate_coo_single_sided(
                    g,
                    feat_compact,
                    el_compact,
                    er_compact,
                    self.leaky_relu_slope,
                    self.compact_direct_indexing_flag,
                )  # NB: kernel modified to enalbe single side
            else:
                raise NotImplementedError(
                    "not implemented the singe side unique node idex"
                )
                h = B.relational_fused_gat_compact_as_of_node(
                    g,
                    feat_compact,
                    el_compact,
                    er_compact,
                    self.leaky_relu_slope,
                )  # TODO: need to modify kernel to enable single side

        else:
            separate_coo_original_dict = g.get_separate_coo_original()
            # with nvtx.annotate("hector_op_category = edgewise mm", color="cyan"):
            feat_src_per_edge = B.rgnn_relational_matmul(
                {
                    "separate_coo_rel_ptrs": separate_coo_original_dict["rel_ptrs"],
                    "separate_coo_node_indices": separate_coo_original_dict[
                        "row_indices"
                    ],
                    "separate_coo_eids": separate_coo_original_dict["eids"],
                },
                self.conv_weights,
                inputs,
                True,
                0,  # CompactAsOfNodeKind::Disabled
            )
            # with nvtx.annotate("hector_op_category = edgewise inner prod", color="cyan"):
            el = B.rgnn_relational_matmul(
                {
                    "separate_coo_rel_ptrs": separate_coo_original_dict["rel_ptrs"],
                    "separate_coo_node_indices": separate_coo_original_dict["eids"],
                    "separate_coo_eids": separate_coo_original_dict["eids"],
                },
                self.attn_l.unsqueeze(-1),
                feat_src_per_edge,
                False,
                0,  # CompactAsOfNodeKind::Disabled
            )

            if self.multiply_among_weights_first_flag:
                separate_coo_original_dict = g.get_separate_coo_original()
                # with nvtx.annotate("hector_op_category = weight mm", color="cyan"):
                product_of_conv_weights_attn_r = th.bmm(
                    self.conv_weights.view(
                        -1, self.in_feat, self.out_feat // self.num_heads
                    ),
                    self.attn_r.view(-1, self.out_feat // self.num_heads, 1),
                ).view(-1, self.num_heads, self.in_feat, 1)
                # with nvtx.annotate("hector_op_category = edgewise (lin op fused) mm", color="cyan"):
                er = B.rgnn_relational_matmul(
                    {
                        "separate_coo_rel_ptrs": separate_coo_original_dict["rel_ptrs"],
                        "separate_coo_node_indices": separate_coo_original_dict[
                            "col_indices"
                        ],
                        "separate_coo_eids": separate_coo_original_dict["eids"],
                    },
                    product_of_conv_weights_attn_r,
                    inputs,
                    False,
                    0,  # CompactAsOfNodeKind::Disabled
                )
            else:
                separate_coo_original_dict = g.get_separate_coo_original()
                feat_dst_per_edge = B.rgnn_relational_matmul(
                    {
                        "separate_coo_rel_ptrs": separate_coo_original_dict["rel_ptrs"],
                        "separate_coo_node_indices": separate_coo_original_dict[
                            "col_indices"
                        ],
                        "separate_coo_eids": separate_coo_original_dict["eids"],
                    },
                    self.conv_weights,
                    inputs,
                    True,
                    0,  # CompactAsOfNodeKind::Disabled
                )
                er = B.rgnn_relational_matmul(
                    {
                        "separate_coo_rel_ptrs": separate_coo_original_dict["rel_ptrs"],
                        "separate_coo_node_indices": separate_coo_original_dict["eids"],
                        "separate_coo_eids": separate_coo_original_dict["eids"],
                    },
                    self.attn_r.unsqueeze(-1),
                    feat_dst_per_edge,
                    False,
                    0,  # CompactAsOfNodeKind::Disabled
                )

            # with nvtx.annotate("hector_op_category = weighted aggregation", color="cyan"):
            if self.gat_edge_parallel_flag:  # NB: use a flag to switch this
                h = B.relational_fused_gat_separate_coo(
                    g, feat_src_per_edge, el, er, self.leaky_relu_slope
                )
            else:
                h = B.relational_fused_gat_csr(
                    g, feat_src_per_edge, el, er, self.leaky_relu_slope
                )

        # NB: let's leverage the built-in bias, activation and dropout here and only focus on SpMM/SDDMM in our kernel implementation.
        # NB: GATConv class also provides bias, activation and dropout but we can ignore them for now.
        # with nvtx.annotate("hector_op_category = activation", color="cyan"):
        h = h.view(-1, self.out_feat)
        if self.self_loop:
            h = h + th.matmul(inputs_dst, self.loop_weight)
        if self.bias:
            h = h + self.h_bias
        if self.activation:
            h = self.activation(h)
        h = self.dropout(h)
        return h


class HET_RelationalGATEncoder(nn.Module):
    # corresponding to EntityClassify in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention encoder

    Parameters
    h_dim: int
        Hidden dimension size
    out_dim: int
        Output dimension size
    num_heads: int
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

    @utils_lite.warn_default_arguments
    def __init__(
        self,
        mydglgraph,
        num_etypes,
        h_dim,
        out_dim,
        num_heads,
        num_hidden_layers=1,
        dropout=0.5,
        use_self_loop: bool = True,
        last_layer_act: bool = False,
        compact_as_of_node_flag: bool = False,
        compact_direct_indexing_flag: bool = False,
        multiply_among_weights_first_flag: bool = False,
        gat_edge_parallel_flag: bool = False,
    ):
        super(HET_RelationalGATEncoder, self).__init__()
        self.mydglgraph = mydglgraph
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.last_layer_act = last_layer_act
        self.compact_as_of_node_flag = compact_as_of_node_flag
        self.compact_direct_indexing_flag = compact_direct_indexing_flag
        self.multiply_among_weights_first_flag = multiply_among_weights_first_flag
        self.gat_edge_parallel_flag = gat_edge_parallel_flag
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
                    self.num_heads,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                    compact_as_of_node_flag=self.compact_as_of_node_flag,
                    compact_direct_indexing_flag=self.compact_direct_indexing_flag,
                    multiply_among_weights_first_flag=self.multiply_among_weights_first_flag,
                    gat_edge_parallel_flag=self.gat_edge_parallel_flag,
                )
            )

        # override the last layer's num_heads to 1 if there are multiple layers, i.e., not in benchmarking
        final_layer_num_heads = self.num_heads if self.num_hidden_layers == 0 else 1
        # h2o
        self.layers.append(
            HET_RelationalAttLayer(
                self.h_dim,
                self.out_dim,
                self.num_etypes,
                final_layer_num_heads,
                activation=F.relu if self.last_layer_act else None,
                self_loop=self.use_self_loop,
                compact_as_of_node_flag=self.compact_as_of_node_flag,
                compact_direct_indexing_flag=self.compact_direct_indexing_flag,
                multiply_among_weights_first_flag=self.multiply_among_weights_first_flag,
                gat_edge_parallel_flag=self.gat_edge_parallel_flag,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
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
                h = layer(self.mydglgraph, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h
