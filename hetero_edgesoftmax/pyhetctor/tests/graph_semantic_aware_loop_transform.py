#!/usr/bin/env python3
for e in g.edges():
    e.dst["sum"] += e.data1
for e in g.edges():
    e["data"] = e.data / e.dst.sum

# transform the code to
# for n in g.nodes():
#     for e in n.incoming_edges():
#         n["sum"]+=e.data1
#     for e in n.incoming_edges():
#         e["data"]=e.data/ n.sum

# then transform it to
# for n in g.nodes():
#     n_sum = 0.0
#     for e in n.incoming_edges():
#         n_sum+=e.data1
#     for e in n.incoming_edges():
#         e["data"]=e.data/ n_sum
