#!/usr/bin/env python3
# adapted from dgl/nn/pytorch/conv/gatconv.py and dgl/nn/pytorch/conv/hgtconv.py
import torch
import torch.nn as nn
from .... import function as fn
from ..linear import TypedLinear
from ..softmax import edge_softmax


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
    ):
        super(GATConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, graph, feat):
        # graph : DGLGraph
        #    The graph.
        # feat : torch.Tensor or pair of torch.Tensor
        #    If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
        #    :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
        #    If a pair of torch.Tensor is given, the pair must contain two tensors of shape
        #    :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            *src_prefix_shape, self._num_heads, self._out_feats
        )
        temp1 = feat_src * self.attn_l
        temp2 = feat_dst * self.attn_r
        el = (temp1).sum(dim=-1).unsqueeze(-1)
        er = (temp2).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({"ft": feat_src, "el": el})
        graph.dstdata.update({"er": er})
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        e = self.leaky_relu(graph.edata.pop("e"))
        temp3 = edge_softmax(graph, e)
        graph.edata["a"] = self.attn_drop(temp3)
