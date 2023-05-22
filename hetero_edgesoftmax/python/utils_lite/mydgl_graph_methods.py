#!/usr/bin/env python3
import torch


# NB: return inverse_idx here so that we can direct index rather than binary search
@torch.no_grad()
def generate_separate_unique_node_idx_single_sided_for_each_etype(
    num_rels, rel_ptr, row_idx, col_idx, get_inverse_idx=False
):
    result_node_idx_row_inverse_idx = []
    result_node_idx_col_inverse_idx = []
    result_node_idx_col = []
    result_rel_ptr_col = [0]
    result_node_idx_row = []
    result_rel_ptr_row = [0]
    for idx_relation in range(num_rels):
        node_idx_for_curr_relation_col = torch.unique(
            col_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
            return_inverse=get_inverse_idx,
        )

        if get_inverse_idx:
            # unpack the result
            (
                node_idx_for_curr_relation_col,
                node_idx_for_curr_relation_col_inverse_idx,
            ) = node_idx_for_curr_relation_col
            node_idx_for_curr_relation_col_inverse_idx = (
                node_idx_for_curr_relation_col_inverse_idx + result_rel_ptr_col[-1]
            )
        else:
            node_idx_for_curr_relation_col_inverse_idx = None

        node_idx_for_curr_relation_row = torch.unique(
            row_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
            return_inverse=get_inverse_idx,
        )
        if get_inverse_idx:
            # unpack the result
            (
                node_idx_for_curr_relation_row,
                node_idx_for_curr_relation_row_inverse_idx,
            ) = node_idx_for_curr_relation_row
            node_idx_for_curr_relation_row_inverse_idx = (
                node_idx_for_curr_relation_row_inverse_idx + result_rel_ptr_row[-1]
            )
        else:
            node_idx_for_curr_relation_row_inverse_idx = None

        result_node_idx_col.append(node_idx_for_curr_relation_col)
        result_rel_ptr_col.append(
            (result_rel_ptr_col[-1] + node_idx_for_curr_relation_col.shape[0])
        )

        result_node_idx_row.append(node_idx_for_curr_relation_row)
        result_rel_ptr_row.append(
            (result_rel_ptr_row[-1] + node_idx_for_curr_relation_row.shape[0])
        )
        # fixme: add bias
        result_node_idx_row_inverse_idx.append(
            node_idx_for_curr_relation_row_inverse_idx
        )
        result_node_idx_col_inverse_idx.append(
            node_idx_for_curr_relation_col_inverse_idx
        )

    result_node_idx_row = torch.concat(result_node_idx_row)
    result_node_idx_col = torch.concat(result_node_idx_col)
    if get_inverse_idx:
        result_node_idx_row_inverse_idx = torch.concat(result_node_idx_row_inverse_idx)
        result_node_idx_col_inverse_idx = torch.concat(result_node_idx_col_inverse_idx)
    else:
        result_node_idx_row_inverse_idx = None
        result_node_idx_col_inverse_idx = None
    result_rel_ptr_col = torch.tensor(result_rel_ptr_col, dtype=torch.int64)
    result_rel_ptr_row = torch.tensor(result_rel_ptr_row, dtype=torch.int64)

    return (
        result_node_idx_row,
        result_rel_ptr_row,
        result_node_idx_col,
        result_rel_ptr_col,
        result_node_idx_row_inverse_idx,
        result_node_idx_col_inverse_idx,
    )


# NB: return inverse_idx here so that we can direct index rather than binary search
@torch.no_grad()
def generate_separate_unique_node_idx_for_each_etype(
    num_rels, rel_ptr, row_idx, col_idx, get_inverse_idx=False
):
    if get_inverse_idx:
        result_node_idx_inverse_idx = []
    result_node_idx = []
    result_rel_ptr = [0]
    for idx_relation in range(num_rels):
        node_idx_for_curr_relation = torch.unique(
            torch.concat(
                [
                    row_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
                    col_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
                ]
            ),
            return_inverse=get_inverse_idx,
        )
        if get_inverse_idx:
            # node_idx_for_curr_relation is a tuple, unpack the result
            (
                node_idx_for_curr_relation,
                node_idx_for_curr_relation_inverse_idx,
            ) = node_idx_for_curr_relation
            result_node_idx_inverse_idx.append(
                node_idx_for_curr_relation_inverse_idx + result_rel_ptr[-1]
            )
        else:
            # node_idx_for_curr_relation is a tensor, do nothing
            pass

        result_node_idx.append(node_idx_for_curr_relation)
        result_rel_ptr.append(
            (result_rel_ptr[-1] + node_idx_for_curr_relation.shape[0])
        )

    result_node_idx = torch.concat(result_node_idx)
    if get_inverse_idx:
        result_node_idx_inverse_idx = torch.concat(result_node_idx_inverse_idx)
    else:
        result_node_idx_inverse_idx = None
    result_rel_ptr = torch.tensor(result_rel_ptr, dtype=torch.int64)
    return result_node_idx, result_rel_ptr, result_node_idx_inverse_idx
