#!/usr/bin/env python3
import torch


# NB: return inverse_indices here so that we can direct index rather than binary search
@torch.no_grad()
def generate_separate_unique_node_indices_single_sided_for_each_etype(
    num_rels, rel_ptrs, row_indices, col_indices, get_inverse_idx=False
):
    result_node_indices_row_inverse_indices = []
    result_node_indices_col_inverse_indices = []
    result_node_indices_col = []
    result_rel_ptrs_col = [0]
    result_node_indices_row = []
    result_rel_ptrs_row = [0]
    for idx_relation in range(num_rels):
        node_indices_for_curr_relation_col = torch.unique(
            col_indices[rel_ptrs[idx_relation] : rel_ptrs[idx_relation + 1]],
            return_inverse=get_inverse_idx,
        )

        if get_inverse_idx:
            # unpack the result
            (
                node_indices_for_curr_relation_col,
                node_indices_for_curr_relation_col_inverse_indices,
            ) = node_indices_for_curr_relation_col
            node_indices_for_curr_relation_col_inverse_indices = (
                node_indices_for_curr_relation_col_inverse_indices
                + result_rel_ptrs_col[-1]
            )
        else:
            node_indices_for_curr_relation_col_inverse_indices = None

        node_indices_for_curr_relation_row = torch.unique(
            row_indices[rel_ptrs[idx_relation] : rel_ptrs[idx_relation + 1]],
            return_inverse=get_inverse_idx,
        )
        if get_inverse_idx:
            # unpack the result
            (
                node_indices_for_curr_relation_row,
                node_indices_for_curr_relation_row_inverse_indices,
            ) = node_indices_for_curr_relation_row
            node_indices_for_curr_relation_row_inverse_indices = (
                node_indices_for_curr_relation_row_inverse_indices
                + result_rel_ptrs_row[-1]
            )
        else:
            node_indices_for_curr_relation_row_inverse_indices = None

        result_node_indices_col.append(node_indices_for_curr_relation_col)
        result_rel_ptrs_col.append(
            (result_rel_ptrs_col[-1] + node_indices_for_curr_relation_col.shape[0])
        )

        result_node_indices_row.append(node_indices_for_curr_relation_row)
        result_rel_ptrs_row.append(
            (result_rel_ptrs_row[-1] + node_indices_for_curr_relation_row.shape[0])
        )
        # fixme: add bias
        result_node_indices_row_inverse_indices.append(
            node_indices_for_curr_relation_row_inverse_indices
        )
        result_node_indices_col_inverse_indices.append(
            node_indices_for_curr_relation_col_inverse_indices
        )

    result_node_indices_row = torch.concat(result_node_indices_row)
    result_node_indices_col = torch.concat(result_node_indices_col)
    if get_inverse_idx:
        result_node_indices_row_inverse_indices = torch.concat(
            result_node_indices_row_inverse_indices
        )
        result_node_indices_col_inverse_indices = torch.concat(
            result_node_indices_col_inverse_indices
        )
    else:
        result_node_indices_row_inverse_indices = None
        result_node_indices_col_inverse_indices = None
    result_rel_ptrs_col = torch.tensor(result_rel_ptrs_col, dtype=torch.int64)
    result_rel_ptrs_row = torch.tensor(result_rel_ptrs_row, dtype=torch.int64)

    return (
        result_node_indices_row,
        result_rel_ptrs_row,
        result_node_indices_col,
        result_rel_ptrs_col,
        result_node_indices_row_inverse_indices,
        result_node_indices_col_inverse_indices,
    )


# NB: return inverse_idx here so that we can direct index rather than binary search
@torch.no_grad()
def generate_separate_unique_node_indices_for_each_etype(
    num_rels, rel_ptrs, row_indices, col_indices, get_inverse_idx=False
):
    if get_inverse_idx:
        result_node_indices_inverse_indices = []
    result_node_indices = []
    result_rel_ptrs = [0]
    for idx_relation in range(num_rels):
        node_indices_for_curr_relation = torch.unique(
            torch.concat(
                [
                    row_indices[rel_ptrs[idx_relation] : rel_ptrs[idx_relation + 1]],
                    col_indices[rel_ptrs[idx_relation] : rel_ptrs[idx_relation + 1]],
                ]
            ),
            return_inverse=get_inverse_idx,
        )
        if get_inverse_idx:
            # node_indices_for_curr_relation is a tuple, unpack the result
            (
                node_indices_for_curr_relation,
                node_indices_for_curr_relation_inverse_indices,
            ) = node_indices_for_curr_relation
            result_node_indices_inverse_indices.append(
                node_indices_for_curr_relation_inverse_indices + result_rel_ptrs[-1]
            )
        else:
            # node_indices_for_curr_relation is a tensor, do nothing
            pass

        result_node_indices.append(node_indices_for_curr_relation)
        result_rel_ptrs.append(
            (result_rel_ptrs[-1] + node_indices_for_curr_relation.shape[0])
        )

    result_node_indices = torch.concat(result_node_indices)
    if get_inverse_idx:
        result_node_indices_inverse_indices = torch.concat(
            result_node_indices_inverse_indices
        )
    else:
        result_node_indices_inverse_indices = None
    result_rel_ptrs = torch.tensor(result_rel_ptrs, dtype=torch.int64)
    return result_node_indices, result_rel_ptrs, result_node_indices_inverse_indices
