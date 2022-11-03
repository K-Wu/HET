#!/usr/bin/env python3
import numpy as np
import torch as th


def get_array_creation_func(torch_flag):
    if torch_flag:
        # return lambda x, dtype: torch.Tensor(x,dtype = dtype)
        return th.tensor
    else:
        return np.array


def get_array_flip_func(torch_flag):
    if torch_flag:
        return lambda x: th.flip(x, dims=[0])
    else:
        return np.flip


def get_array_concatenate_func(torch_flag):
    if torch_flag:
        return th.cat
    else:
        return np.concatenate


def remap_etype_according_to_number_of_edges(etypes, torch_flag):
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_flip_func = get_array_flip_func(torch_flag)
    array_creation_func = get_array_creation_func(torch_flag)
    # expecting numpy array
    etype_frequency = np_or_th.bincount(etypes.flatten())
    # TODO: check if there is data loss in this np.argsort invocation, i.e., implicit conversion from int64 to int32
    etype_sorted_by_frequency_from_largest_to_smallest = array_flip_func(
        np_or_th.argsort(etype_frequency)
    )
    original_etype_to_new_etype_map = dict(
        zip(
            etype_sorted_by_frequency_from_largest_to_smallest.tolist(),
            range(len(etype_sorted_by_frequency_from_largest_to_smallest)),
        )
    )
    remapped_etype = array_creation_func(
        [original_etype_to_new_etype_map[etype] for etype in etypes], dtype=etypes.dtype
    )
    return remapped_etype


def get_node_index_remap_dict_according_to_number_of_edges(
    srcs, number_of_nodes, torch_flag
):
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_flip_func = get_array_flip_func(torch_flag)
    array_concatenate_func = get_array_concatenate_func(torch_flag)

    srcs_frequency = np_or_th.bincount(srcs.flatten())
    if srcs_frequency.shape[0] < number_of_nodes:
        srcs_frequency = array_concatenate_func(
            [srcs_frequency, np_or_th.zeros(number_of_nodes - srcs_frequency.shape[0])]
        )
    # TODO: check if there is data loss in this np.argsort invocation, i.e., implicit conversion from int64 to int32
    original_src_to_new_src_map = dict(
        zip(
            array_flip_func(np_or_th.argsort(srcs_frequency)).tolist(),
            range(number_of_nodes),
        )
    )
    return original_src_to_new_src_map


def remap_node_indices_according_to_number_of_edges(srcs, dests, torch_flag):
    # if torch_flag:
    #     np_or_th = th
    # else:
    #     np_or_th = np
    # array_flip_func = get_array_flip_func(torch_flag)
    array_creation_func = get_array_creation_func(torch_flag)
    original_src_to_new_src_map = (
        get_node_index_remap_dict_according_to_number_of_edges(
            srcs, max(srcs.max(), dests.max()) + 1, torch_flag
        )
    )
    remapped_src = array_creation_func(
        [original_src_to_new_src_map[src] for src in srcs], dtype=srcs.dtype
    )
    remapped_dest = array_creation_func(
        [original_src_to_new_src_map[dest] for dest in dests], dtype=dests.dtype
    )
    return remapped_src, remapped_dest


def sort_coo_by_etype(srcs, dsts, etypes, eids, torch_flag=False, infidel_flag=False):
    print(
        "WARNING: you are using infidel sort utility. See readme.md for more details."
    )
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_creation_func = get_array_creation_func(torch_flag)
    etypes_priority = remap_etype_according_to_number_of_edges(etypes, torch_flag)
    # now sort the (src, dst) pair according to their etype
    sorted_src_dst_etype = sorted(
        zip(srcs, dsts, etypes, eids, etypes_priority), key=lambda x: x[4]
    )
    sorted_srcs = array_creation_func([x[0] for x in sorted_src_dst_etype])
    sorted_dsts = array_creation_func([x[1] for x in sorted_src_dst_etype])
    if infidel_flag:
        sorted_etypes = array_creation_func([x[2] for x in sorted_src_dst_etype])
    else:
        sorted_etypes = array_creation_func([x[4] for x in sorted_src_dst_etype])
    sorted_eids = array_creation_func([x[3] for x in sorted_src_dst_etype])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids


def sort_coo_by_src_outgoing_edges(
    srcs, dsts, etypes, eids, torch_flag=False, infidel_flag=False
):
    print(
        "WARNING: you are using infidel sort utility. See readme.md for more details."
    )
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_creation_func = get_array_creation_func(torch_flag)
    srcs, _ = remap_node_indices_according_to_number_of_edges(srcs, torch_flag)
    if not infidel_flag:
        dsts = _
    # now sort the (src, dst) pair according to their src idx
    sorted_src_dst_srcs = sorted(zip(srcs, dsts, etypes, eids), key=lambda x: x[0])
    sorted_srcs = array_creation_func([x[0] for x in sorted_src_dst_srcs])
    sorted_dsts = array_creation_func([x[1] for x in sorted_src_dst_srcs])
    sorted_etypes = array_creation_func([x[2] for x in sorted_src_dst_srcs])
    sorted_eids = array_creation_func([x[3] for x in sorted_src_dst_srcs])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids
