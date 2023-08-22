#!/usr/bin/env python3
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import torch as th
import dgl


# This is the data preparation logic from the original RGAT script. Keeping this function for compatibility.
def _legacy_RGAT_prepare_mag_data(args: argparse.Namespace):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    split_idx = dataset.get_idx_split()
    g, labels = dataset[
        0
    ]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    labels = labels["paper"].flatten()

    def add_reverse_hetero(g, combine_like: bool = True):
        r"""
        Parameters
        ----------
        g : DGLGraph
            The heterogenous graph where reverse edges should be added
        combine_like : bool, optional
            Whether reverse-edges that have identical source/destination
            node types should be combined with the existing edge-type,
            rather than creating a new edge type.  Default: True.
        """
        relations = {}
        num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        for metapath in g.canonical_etypes:
            src_ntype, rel_type, dst_ntype = metapath
            src, dst = g.all_edges(etype=rel_type)

            if src_ntype == dst_ntype and combine_like:
                # Make edges un-directed instead of making a reverse edge type
                relations[metapath] = (
                    th.cat([src, dst], dim=0),
                    th.cat([dst, src], dim=0),
                )
            else:
                # Original edges
                relations[metapath] = (src, dst)

                reverse_metapath = (dst_ntype, "rev-" + rel_type, src_ntype)
                relations[reverse_metapath] = (dst, src)  # Reverse edges

        new_g = dgl.heterograph(relations, num_nodes_dict=num_nodes_dict)
        # Remove duplicate edges
        new_g = dgl.to_simple(
            new_g, return_counts=None, writeback_mapping=False, copy_ndata=True
        )

        # copy_ndata:
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[k] = v.detach().clone()

        return new_g

    g = add_reverse_hetero(g)
    print("Loaded graph: {}".format(g))

    # logger = Logger(args['runs'], args)

    # train sampler
    if args.full_graph_training:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            args.num_layers)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
    train_loader = dgl.dataloading.NodeDataLoader(
        g,
        split_idx["train"],
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    return (g, labels, dataset.num_classes, split_idx, train_loader)
