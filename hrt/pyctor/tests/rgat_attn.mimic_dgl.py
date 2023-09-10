#!/usr/bin/env python3

# adapted from dgl/nn/pytorch/conv/gatconv.py and dgl/nn/pytorch/conv/hgtconv.py
import torch
import torch.nn as nn
from .... import function as fn
from ..linear import TypedLinear
from ..softmax import edge_softmax


class RGATConv(nn.Module):
    def __init__(
        self,
        in_size,
        head_size,
        num_heads,
        num_etypes,
        dropout=0.2,
        use_norm=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.use_norm = use_norm

        self.linear_w = TypedLinear(in_size, head_size * num_heads, num_etypes)
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(num_etypes, num_heads, head_size))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(num_etypes, num_heads, head_size))
        )

    def forward(self, graph, x, etype, *, presorted=True):
        # graph : DGLGraph a homogeneous graph with type info encoded as separate array
        #    The input graph.
        # x : torch.Tensor
        #    A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        # ntype : torch.Tensor
        #    An 1D integer tensor of node types. Shape: :math:`(|V|,)`.
        # etype : torch.Tensor
        #    An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        feat_src = self.linear_w(x, etype, presorted=presorted).view(
            -1, self.num_heads, self.head_size
        )
        feat_dst = self.linear_w(x, etype, presorted=presorted).view(
            -1, self.num_heads, self.head_size
        )
        attn_l = self.attn_l.index_select(0, etype).view(
            -1, self.num_heads, self.head_size
        )
        attn_r = self.attn_r.index_select(0, etype).view(
            -1, self.num_heads, self.head_size
        )
        temp1 = torch.bmm(feat_src, attn_l)
        temp2 = torch.bmm(feat_dst, attn_r)
        el = (temp1).sum(dim=-1).unsqueeze(-1)
        er = (temp2).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({"ft": feat_src, "el": el})
        graph.dstdata.update({"er": er})
        graph.apply_edges(fn.u_add_v("el", "er", "e"))
        # graph.apply_edges(self.message)
        e = self.leaky_relu(graph.edata.pop("e"))
        temp3 = edge_softmax(graph, e)
        graph.edata["a"] = self.attn_drop(temp3)

    # def message(self, edges):
    #     """Message function."""
    #     with nvtx.annotate("hector_op_category = mm", color="cyan"):
    #         a, m = [], []
    #         etype = edges.data['etype']
    #         k = torch.unbind(edges.src['k'], dim=1)
    #         q = torch.unbind(edges.dst['q'], dim=1)
    #         v = torch.unbind(edges.src['v'], dim=1)
    #         for i in range(self.num_heads):
    #             kw = self.relation_att[i](k[i], etype, self.presorted)  # (E, O)
    #             a.append((kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d)  # (E,)
    #             m.append(self.relation_msg[i](v[i], etype, self.presorted))  # (E, O)
    #     return {'a': torch.stack(a, dim=1), 'm': torch.stack(m, dim=1)}
