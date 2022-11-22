import argparse
from .models import (
    HET_RelationalGATEncoder,
    HET_RelationalAttLayer,
)
from .models_dgl import (
    RelationalGATEncoder,
    RelationalAttLayer,
    RelGraphEmbed,
    HET_RelGraphEmbed,
)
import dgl
import argparse
import itertools
import torch as th
from torch import nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from .. import utils


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb


def RGAT_parse_args() -> argparse.Namespace:
    # DGL
    parser = argparse.ArgumentParser(description="RGAT")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument("-d", "--dataset", type=str, default="ogbn-mag", help="dataset")

    parser.add_argument(
        "--n_hidden", type=int, default=64, help="number of hidden units"
    )
    parser.add_argument(
        "--sparse_format", type=str, default="csr", help="sparse format to use"
    )
    parser.add_argument("--sort_by_src", action="store_true", help="sort by src")
    parser.add_argument("--sort_by_etype", action="store_true", help="sort by etype")
    parser.add_argument(
        "--reindex_eid",
        action="store_true",
        help="use new eid after sorting rather than load referential eids",
    )
    parser.add_argument("--num_classes", type=int, default=8, help="number of classes")
    parser.add_argument(
        "--use_real_labels_and_features", action="store_true", help="use real labels"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n_head", type=int, default=2, help="number of heads")
    parser.add_argument(
        "-e", "--n_epochs", type=int, default=3, help="number of training epochs"
    )
    parser.add_argument("--fanout", type=int, nargs="+", default=[25, 20])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--full_graph_training", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--compact_as_of_node_flag", action="store_true")
    # OGB
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()
    return args


def RGAT_get_model(g, num_classes, hypermeters):
    embed_layer = RelGraphEmbed(
        g, hypermeters["n_hidden"], exclude=[]
    )  # exclude=["paper"])

    model = RelationalGATEncoder(
        g,
        h_dim=hypermeters["n_hidden"],
        out_dim=num_classes,
        n_heads=hypermeters["n_head"],
        num_hidden_layers=hypermeters["num_layers"] - 1,
        dropout=hypermeters["dropout"],
        use_self_loop=True,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def RGAT_get_our_model(
    g: utils.MyDGLGraph, num_classes, args: argparse.Namespace
) -> tuple[HET_RelGraphEmbed, HET_RelationalGATEncoder]:
    embed_layer = HET_RelGraphEmbed(g, args.n_hidden, exclude=[])  # exclude=["paper"])

    model = HET_RelationalGATEncoder(
        g.get_num_rels(),
        h_dim=args.n_hidden,
        out_dim=num_classes,
        n_heads=args.n_head,
        num_hidden_layers=args.num_layers - 1,
        dropout=args.dropout,
        use_self_loop=True,
        compact_as_of_node_flag=args.compact_as_of_node_flag,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def prepare_data(args: argparse.Namespace):
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
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
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


def HET_RGAT_train_with_sampler(
    g, model, node_embed, optimizer, train_loader, labels, device, hypermeters
):
    raise NotImplementedError("HET_RGAT_train_with_sampler not implemented yet")


def HET_RGAT_train_full_graph(
    g: utils.MyDGLGraph,
    model,
    node_embed_layer,
    optimizer,
    labels: th.Tensor,
    device,
    hypermeters: dict,
):
    # training loop
    print("start training...")
    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()
        model.requires_grad_(True)
        node_embed_layer.train()
        node_embed_layer.requires_grad_(True)

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )

        # emb = extract_embed(node_embed, input_nodes)
        # emb = node_embed

        # Add the batch's raw "paper" features
        # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})
        node_embed = node_embed_layer()

        optimizer.zero_grad()
        forward_prop_start = th.cuda.Event(enable_timing=True)
        forward_prop_end = th.cuda.Event(enable_timing=True)
        forward_prop_start.record()
        logits = model(g, node_embed)
        # logits = model(emb, blocks)

        y_hat = logits.log_softmax(dim=-1)
        loss = F.nll_loss(y_hat, labels)
        forward_prop_end.record()
        th.cuda.synchronize()

        backward_prop_start = th.cuda.Event(enable_timing=True)
        backward_prop_end = th.cuda.Event(enable_timing=True)
        backward_prop_start.record()
        # loss.backward()

        # optimizer.step()
        backward_prop_end.record()
        th.cuda.synchronize()

        # FIXME: should be # edges when training full graph
        total_loss += loss.item() * hypermeters["batch_size"]

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
        print(
            f"Forward prop time: {forward_prop_start.elapsed_time(forward_prop_end)} ms"
        )
        print(
            f"Backward prop time: {backward_prop_start.elapsed_time(backward_prop_end)} ms"
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGAT_train_full_graph(model, node_embed, optimizer, labels, hypermeters: dict):
    # training loop
    print("start training...")
    category = "paper"
    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )

        # emb = extract_embed(node_embed, input_nodes)
        emb = node_embed

        # Add the batch's raw "paper" features
        # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})

        lbl = labels

        if th.cuda.is_available():
            emb = {k: e.cuda() for k, e in emb.items()}
            lbl = {k: e.cuda() for k, e in lbl.items()}

        optimizer.zero_grad()
        forward_prop_start = th.cuda.Event(enable_timing=True)
        forward_prop_end = th.cuda.Event(enable_timing=True)
        forward_prop_start.record()
        logits = model(emb)
        # logits = model(emb, blocks)
        loss = None
        for category in logits:
            y_hat = logits[category].log_softmax(dim=-1)
            if loss is None:
                loss = F.nll_loss(y_hat, lbl)
            else:
                loss += F.nll_loss(y_hat, lbl)
        forward_prop_end.record()
        th.cuda.synchronize()

        backward_prop_start = th.cuda.Event(enable_timing=True)
        backward_prop_end = th.cuda.Event(enable_timing=True)
        backward_prop_start.record()
        loss.backward()
        optimizer.step()
        backward_prop_end.record()
        th.cuda.synchronize()

        # FIXME: should be # edges when training full graph
        total_loss += loss.item() * args.batch_size

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
        print(
            f"Forward prop time: {forward_prop_start.elapsed_time(forward_prop_end)} ms"
        )
        print(
            f"Backward prop time: {backward_prop_start.elapsed_time(backward_prop_end)} ms"
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


# * g(dglgraph) is already set as a member of model
def RGAT_train_with_sampler(
    model, node_embed, optimizer, train_loader, labels, device, hypermeters: dict
):
    # training loop
    print("start training...")

    for epoch in range(hypermeters["n_epochs"]):

        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )
        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]
            # seeds = seeds[category]  # we only predict the nodes with type "category"

            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features
            # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})

            # lbl = th.concat([labels[seeds[category]] for category in seeds])
            # lbl = labels[seeds]
            lbl = labels

            if th.cuda.is_available():
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()

            optimizer.zero_grad()
            if 0:
                # the following is the original code. Keep it for reference
                category = "paper"
                logits = model(emb, blocks)[category]
                # logits = model(emb, blocks)

                y_hat = logits.log_softmax(dim=-1)
                # y_hat = th.concat([logits[category][seeds[category]] for category in logits if category in seeds]).log_softmax(dim=-1)
                loss = F.nll_loss(y_hat, lbl)
            else:
                logits = model(emb, blocks)
                # logits = model(emb, blocks)
                loss = None
                for category in logits:
                    if category in seeds:
                        y_hat = logits[category].log_softmax(dim=-1)
                        if loss is None:
                            loss = F.nll_loss(y_hat, lbl[seeds[category]])
                        else:
                            loss += F.nll_loss(y_hat, lbl[seeds[category]])

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * args.batch_size
            # pbar.update(batch_size)

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Epoch: {epoch + 1 :02d}, " f"Loss (w/o dividing sample num): {loss:.4f}, "
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


def RGAT_main_procedure(args: argparse.Namespace, dgl_model_flag: bool):

    # Static parameters
    # NB: default values are all moved to args
    # hyperparameters = dict(
    # num_layers=2,
    # fanout=[25, 20],
    # batch_size=1024,
    # )
    # hyperparameters.update(vars(args))
    hyperparameters = vars(args)
    print(hyperparameters)
    if not args.full_graph_training:
        assert len(args.fanout) == args.num_layers
    device = f"cuda:0" if th.cuda.is_available() else "cpu"
    # loading data
    if dgl_model_flag:
        if args.dataset != "ogbn-mag":
            raise NotImplementedError(
                "Only ogbn-mag dataset is supported for dgl model."
            )
        (g, labels, num_classes, split_idx, train_loader) = prepare_data(args)
    else:
        # (g, labels, num_classes, split_idx, train_loader) = prepare_data(hyperparameters)
        g = utils.RGNN_get_mydgl_graph(
            args.dataset,
            args.sort_by_src,
            args.sort_by_etype,
            args.reindex_eid,
            args.sparse_format,
        )
        if args.use_real_labels_and_features:
            raise NotImplementedError(
                "Not implemented loading real labels and features in utils.RGNN_get_mydgl_graph"
            )

    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    # creating model
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        if dgl_model_flag:
            labels = th.randint(0, args.num_classes, labels.shape)
        else:
            print(
                "WARNING: assuming node classification in RGAT_main_procedure(dgl_model_flag == False)"
            )
            labels = th.randint(0, args.num_classes, [g.get_num_nodes()])
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = RGAT_get_model(g, num_classes, hyperparameters)
    else:
        print("Using our RGAT model")
        # print(
        # int(g["original"]["col_idx"].max()) + 1,
        # )
        # print(g["original"]["row_ptr"].numel() - 1)
        embed_layer, model = RGAT_get_our_model(g, num_classes, args)
        # TODO: only certain design choices call for this. Add an option to choose.

        g.generate_separate_coo_adj_for_each_etype(transposed_flag=True)
        g.generate_separate_coo_adj_for_each_etype(transposed_flag=False)
        g.get_separate_node_idx_for_each_etype()
        if not args.full_graph_training:
            # need to prepare dgl graph for sampler
            g_dglgraph = g.get_dgl_graph()
            # train sampler
            # TODO: figure out split_idx train for this case
            sampler = dgl.dataloading.MultiLayerNeighborSampler(args.fanout)
            train_loader = dgl.dataloading.NodeDataLoader(
                g_dglgraph,
                split_idx["train"],
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
            )
        g = g.to(device)

    embed_layer = embed_layer.to(device)
    model = model.to(device)
    labels = labels.to(device)
    for run in range(args.runs):
        embed_layer.reset_parameters()
        model.reset_parameters()

        # optimizer
        all_params = [*model.parameters()] + [
            *embed_layer.parameters()
        ]  # itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr)
        print(f"Run: {run + 1:02d}, ")
        if dgl_model_flag:
            if args.full_graph_training:
                RGAT_train_full_graph(
                    model,
                    embed_layer,
                    labels,
                    # device,
                    optimizer,
                    hyperparameters,
                )
            else:
                RGAT_train_with_sampler(
                    model,
                    embed_layer(),
                    optimizer,
                    train_loader,
                    labels,
                    device,
                    hyperparameters,
                )
        else:
            if not args.full_graph_training:
                raise NotImplementedError(
                    "Not implemented full_graph_training in RGAT_main_procedure(dgl_model_flag == False)"
                )
            HET_RGAT_train_full_graph(
                g,
                model,
                embed_layer,
                optimizer,
                labels,
                device,
                hyperparameters,
            )
        # logger.print_statistics(run)

    # print("Final performance: ")
    # logger.print_statistics()


if __name__ == "__main__":
    args: argparse.Namespace = RGAT_parse_args()
    print(args)
    RGAT_main_procedure(args, dgl_model_flag=True)
