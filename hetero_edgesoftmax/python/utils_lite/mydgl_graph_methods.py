#!/usr/bin/env python3
import torch


# TODO: return reverse_idx if needed
@torch.no_grad()
def generate_get_separate_unique_node_idx_single_sided_for_each_etype(
    rel_ptr, row_idx, col_idx
):
    result_node_idx_col = []
    result_rel_ptr_col = [0]
    result_node_idx_row = []
    result_rel_ptr_row = [0]
    for idx_relation in range(self.get_num_rels()):
        node_idx_for_curr_relation_col = torch.unique(
            col_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]]
        )
        node_idx_for_curr_relation_row = torch.unique(
            row_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]]
        )

        result_node_idx_col.append(node_idx_for_curr_relation_col)
        result_rel_ptr_col.append(
            (result_rel_ptr_col[-1] + node_idx_for_curr_relation_col.shape[0])
        )

        result_node_idx_row.append(node_idx_for_curr_relation_row)
        result_rel_ptr_row.append(
            (result_rel_ptr_row[-1] + node_idx_for_curr_relation_row.shape[0])
        )

    result_node_idx_row = torch.concat(result_node_idx_row)
    result_node_idx_col = torch.concat(result_node_idx_col)
    result_rel_ptr_col = torch.tensor(result_rel_ptr_col, dtype=torch.int64)
    result_rel_ptr_row = torch.tensor(result_rel_ptr_row, dtype=torch.int64)

    return (
        result_node_idx_row,
        result_rel_ptr_row,
        result_node_idx_col,
        result_rel_ptr_col,
    )


# TODO: return reverse_idx if needed
@torch.no_grad()
def generate_separate_unique_node_idx_for_each_etype(rel_ptr, row_idx, col_idx):
    result_node_idx = []
    result_rel_ptr = [0]
    for idx_relation in range(self.get_num_rels()):
        node_idx_for_curr_relation = torch.unique(
            torch.concat(
                [
                    row_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
                    col_idx[rel_ptr[idx_relation] : rel_ptr[idx_relation + 1]],
                ]
            )
        )

        result_node_idx.append(node_idx_for_curr_relation)
        result_rel_ptr.append(
            (result_rel_ptr[-1] + node_idx_for_curr_relation.shape[0])
        )

    result_node_idx = torch.concat(result_node_idx)
    result_rel_ptr = torch.tensor(result_rel_ptr, dtype=torch.int64)
    return result_node_idx, result_rel_ptr
