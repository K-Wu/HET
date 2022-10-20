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


def remap_src_according_to_number_of_edges(srcs, torch_flag):
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_flip_func = get_array_flip_func(torch_flag)
    array_creation_func = get_array_creation_func(torch_flag)

    srcs_frequency = np_or_th.bincount(srcs.flatten())
    # TODO: check if there is data loss in this np.argsort invocation, i.e., implicit conversion from int64 to int32
    srcs_sorted_by_frequency_from_largest_to_smallest = array_flip_func(
        np_or_th.argsort(srcs_frequency)
    )
    original_src_to_new_src_map = dict(
        zip(
            srcs_sorted_by_frequency_from_largest_to_smallest.tolist(),
            range(len(srcs_sorted_by_frequency_from_largest_to_smallest)),
        )
    )
    remapped_src = array_creation_func(
        [original_src_to_new_src_map[src] for src in srcs], dtype=srcs.dtype
    )
    return remapped_src


def infidel_sort_coo_by_etype(srcs, dsts, etypes, eids, torch_flag=False):
    print(
        "WARNING: you are using infidel sort utility. See readme.md for more details."
    )
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_creation_func = get_array_creation_func(torch_flag)
    etypes = remap_etype_according_to_number_of_edges(etypes, torch_flag)
    # now sort the (src, dst) pair according to their etype
    sorted_src_dst_etype = sorted(zip(srcs, dsts, etypes, eids), key=lambda x: x[2])
    sorted_srcs = array_creation_func([x[0] for x in sorted_src_dst_etype])
    sorted_dsts = array_creation_func([x[1] for x in sorted_src_dst_etype])
    sorted_etypes = array_creation_func([x[2] for x in sorted_src_dst_etype])
    sorted_eids = array_creation_func([x[3] for x in sorted_src_dst_etype])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids


def infidel_sort_coo_by_src_outgoing_edges(srcs, dsts, etypes, eids, torch_flag=False):
    print(
        "WARNING: you are using infidel sort utility. See readme.md for more details."
    )
    if torch_flag:
        np_or_th = th
    else:
        np_or_th = np
    array_creation_func = get_array_creation_func(torch_flag)
    srcs = remap_src_according_to_number_of_edges(srcs, torch_flag)
    # now sort the (src, dst) pair according to their src idx
    sorted_src_dst_srcs = sorted(zip(srcs, dsts, etypes, eids), key=lambda x: x[0])
    sorted_srcs = array_creation_func([x[0] for x in sorted_src_dst_srcs])
    sorted_dsts = array_creation_func([x[1] for x in sorted_src_dst_srcs])
    sorted_etypes = array_creation_func([x[2] for x in sorted_src_dst_srcs])
    sorted_eids = array_creation_func([x[3] for x in sorted_src_dst_srcs])
    return sorted_srcs, sorted_dsts, sorted_etypes, sorted_eids
