#!/usr/bin/env python3
# TODO: use dgl.sampling.neighbor._CAPI_DGLSampleNeighbors as examplified in dgl/python/dgl/sampling/neighbor.py
# TODO: use subgraph/block.ndata[dgl.NID] and subgraph/block.edata[dgl.EID] to retrieve original node/edge IDs. Source: https://docs.dgl.ai/en/0.9.x/generated/dgl.node_subgraph.html

import dgl
import torch

# hack old pytorch where concatenation is provided as torch.hstack
if "concat" not in torch.__dict__:
    torch.concat = torch.hstack
from ..utils_lite import coo2csr


def extract_adj_arrays_from_dgl_subgraph(
    parent_mydgl_graph,
    block,
    sparse_format,
    etype_ptr_instead_of_id_flag: bool = False,
    transpose_flag: bool = False,
):
    # block being either DGLBlock or DGLSubblock should be fine
    subgraph_etype_set = set(block.canonical_etypes)

    parent_nid = block.ndata[dgl.NID]
    parent_eid = block.edata[dgl.EID]
    subgraph_coos_row_indices_or_pointers = []
    subgraph_coos_col_indices = []
    subgraph_coos_eids = []
    subgraph_coos_etypes = []
    subgraph_coos_etype_element_num = []

    for idx_etype, etype in enumerate(parent_mydgl_graph.canonical_etypes):
        if (
            etype not in subgraph_etype_set
            or block.number_of_edges(etype=etype) == 0
        ):
            if etype_ptr_instead_of_id_flag:
                subgraph_coos_etype_element_num.append(0)
            continue
        curr_subgraph_row_indices_before_index_remap = block.edges(
            etype=etype
        )[0]
        curr_subgraph_col_indices_before_index_remap = block.edges(
            etype=etype
        )[1]

        if transpose_flag:
            (
                curr_subgraph_row_indices_before_index_remap,
                curr_subgraph_col_indices_before_index_remap,
            ) = (
                curr_subgraph_col_indices_before_index_remap,
                curr_subgraph_row_indices_before_index_remap,
            )

        curr_subgraph_eids = parent_eid[etype]
        curr_subgraph_row_indices = parent_nid[etype][0][
            curr_subgraph_row_indices_before_index_remap
        ]
        curr_subgraph_col_indices = parent_nid[etype][1][
            curr_subgraph_col_indices_before_index_remap
        ]
        curr_subgraph_etypes = torch.tensor(
            [idx_etype] * len(curr_subgraph_row_indices), dtype=torch.int64
        )

        if sparse_format == "coo":
            subgraph_coos_row_indices_or_pointers.append(
                curr_subgraph_row_indices
            )
        elif sparse_format == "csr":
            (
                curr_subgraph_row_ptrs,
                curr_subgraph_col_indices,
                curr_subgraph_etypes,
                curr_subgraph_eids,
            ) = coo2csr(
                curr_subgraph_row_indices,
                curr_subgraph_col_indices,
                curr_subgraph_etypes,
                curr_subgraph_eids,
                torch_flag=True,
            )
            subgraph_coos_row_indices_or_pointers.append(
                curr_subgraph_row_ptrs
            )
        else:
            raise ValueError("sparse_format must be either coo or csr")

        subgraph_coos_col_indices.append(curr_subgraph_col_indices)
        subgraph_coos_eids.append(curr_subgraph_eids)
        if etype_ptr_instead_of_id_flag:
            subgraph_coos_etype_element_num.append(len(curr_subgraph_eids))
        else:
            subgraph_coos_etypes.append(curr_subgraph_etypes)

    if etype_ptr_instead_of_id_flag:
        subgraph_rel_ptr = torch.tensor(
            [0] + subgraph_coos_etype_element_num, dtype=torch.int64
        ).cumsum(dim=0)
        return (
            torch.concat(subgraph_coos_row_indices_or_pointers).to(
                device=parent_mydgl_graph.get_device()
            ),
            torch.concat(subgraph_coos_col_indices).to(
                device=parent_mydgl_graph.get_device()
            ),
            torch.concat(subgraph_coos_eids).to(
                device=parent_mydgl_graph.get_device()
            ),
            subgraph_rel_ptr.to(device=parent_mydgl_graph.get_device()),
        )
    else:
        return (
            torch.concat(subgraph_coos_row_indices_or_pointers).to(
                device=parent_mydgl_graph.get_device()
            ),
            torch.concat(subgraph_coos_col_indices).to(
                device=parent_mydgl_graph.get_device()
            ),
            torch.concat(subgraph_coos_eids).to(
                device=parent_mydgl_graph.get_device()
            ),
            torch.concat(subgraph_coos_etypes).to(
                device=parent_mydgl_graph.get_device()
            ),
        )
