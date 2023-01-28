#!/usr/bin/env python3
for e in g.edges():
    zi = e.src.feature * W[e.etype]
    zj = e.dst.feature * W[e.etype]
    e["attn"] = leakyrelu(inner_prod(attn_vec[e.etype], concat([zi, zj])))
    e["attn"] = exp(e["attn"])
for n in g.dst_nodes():
    n["attn_sum"] = 0.0
    for e in n.incoming_edges():
        n["attn_sum"] += e["attn"]
for e in g.edges():
    e["attn"] = e["attn"] / e.dst["attn_sum"]
