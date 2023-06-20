#!/usr/bin/env python3
import ast


# a list contains all matcher functions defined in this file
matchers = []

# TODO: there might be a chain of operations in one line, e.g.,
# n.output = n.feature * transpose(W)
# TO deal with this, we need to do the match in the following steps after canonicalization pass
# 1. match assignment. This helps us figure out the output
# 2. recursively match the right hand side of the assignment, where the right hand side entry function will be finally called at the end of any match function, i.e., after all other non-chain match logic failed
# note that upon entering the right-hand-side matching function, pass in 1) left-hand side results, and 2) for-loop levels. During recursive call, we can pass in the temporary result name instead. For example.
# when matchiing n.output = n.feature * transpose(W), the first time right-hand-side match is called, "node-wise iteration" and "n.output" are passed in. The second time right-hand-side match is called, "node-wise iteration" and "n.output_tmp1" are passed in
