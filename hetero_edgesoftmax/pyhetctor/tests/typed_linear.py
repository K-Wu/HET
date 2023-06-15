#!/usr/bin/env python3

linear_k = TypedLinear(in_size, head_size * num_heads, num_ntypes)
# linear_k is Typedlinear
zi = linear_k(x, ntype, presorted)
