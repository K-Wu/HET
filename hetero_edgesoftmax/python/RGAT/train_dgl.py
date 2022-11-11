import argparse
from .models import (
    HET_RelationalGATEncoder,
    HET_RelationalAttLayer,
    HET_RelationalGATEncoderSingleLayer,
)
from .models_dgl import RelationalGATEncoder, RelationalAttLayer, RelGraphEmbed
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


def RGAT_parse_args():
    # DGL
    parser = argparse.ArgumentParser(description="RGAT")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=64, help="number of hidden units"
    )
    parser.add_argument("--num_classes", type=int, default=8, help="number of classes")
    parser.add_argument(
        "--use_real_labels_and_features", action="store_true", help="use real labels"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-head", type=int, default=2, help="number of heads")
    parser.add_argument(
        "-e", "--n-epochs", type=int, default=3, help="number of training epochs"
    )

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


def RGAT_get_our_model(g, num_classes, args):
    embed_layer = RelGraphEmbed(g, args.n_hidden, exclude=[])  # exclude=["paper"])

    model = HET_RelationalGATEncoder(
        g,
        h_dim=args.n_hidden,
        out_dim=num_classes,
        n_heads=args.n_head,
        num_hidden_layers=args.num_layers - 1,
        dropout=args.dropout,
        use_self_loop=True,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


def prepare_data(args):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    split_idx = dataset.get_idx_split()
    g, labels = dataset[
        0
    ]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    labels = labels["paper"].flatten()

    def add_reverse_hetero(g, combine_like=True):
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
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args["fanout"])
    train_loader = dgl.dataloading.NodeDataLoader(
        g,
        split_idx["train"],
        sampler,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    return (g, labels, dataset.num_classes, split_idx, train_loader)


def train(
    g, model, node_embed, optimizer, train_loader, split_idx, labels, device, run, args
):
    # training loop
    print("start training...")
    category = "paper"

    for epoch in range(args["n_epochs"]):
        N_train = split_idx["train"][category].shape[0]
        print(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0
        print(
            "WARNING: ignoring the hard-coded paper features in the original dataset. This script is solely for performance R&D purposes."
        )
        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[category]  # we only predict the nodes with type "category"
            batch_size = seeds.shape[0]

            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features
            # emb.update({"paper": g.ndata["feat"]["paper"][input_nodes["paper"]]})

            lbl = labels[seeds]

            if th.cuda.is_available():
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()

            optimizer.zero_grad()
            logits = model(emb, blocks)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            # pbar.update(batch_size)

        loss = total_loss / N_train

        # result = test(g, model, node_embed, labels, device, split_idx, args)
        # logger.add_result(run, result)
        # train_acc, valid_acc, test_acc = result
        print(
            f"Run: {run + 1:02d}, " f"Epoch: {epoch + 1 :02d}, " f"Loss: {loss:.4f}, "
        )
    #              f'Train: {100 * train_acc:.2f}%, '
    #              f'Valid: {100 * valid_acc:.2f}%, '
    #              f'Test: {100 * test_acc:.2f}%')

    return  # logger


def RGAT_main_procedure(args, dgl_model_flag):
    # Static parameters
    hyperparameters = dict(
        # num_layers=2,
        fanout=[25, 20],
        batch_size=1024,
    )
    hyperparameters.update(vars(args))
    print(hyperparameters)

    device = f"cuda:0" if th.cuda.is_available() else "cpu"

    (g, labels, num_classes, split_idx, train_loader) = prepare_data(hyperparameters)
    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    if not args.use_real_labels_and_features:
        num_classes = args.num_classes
        labels = th.randint(0, args.num_classes, labels.shape)
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = RGAT_get_model(g, num_classes, hyperparameters)
    else:
        g = utils.create_mydgl_graph_coo_from_dgl_graph(g)
        print("Using our RGAT model")
        embed_layer, model = RGAT_get_our_model(g, num_classes, args)
        g.get_separate_node_idx_for_each_etype()
        g.generate_separate_coo_adj_for_each_etype()
        g = g.to(device)
    model = model.to(device)

    for run in range(args.runs):
        embed_layer.reset_parameters()
        model.reset_parameters()

        # optimizer
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr)

        train(
            g,
            model,
            embed_layer(),
            optimizer,
            train_loader,
            split_idx,
            labels,
            device,
            run,
            hyperparameters,
        )

        # logger.print_statistics(run)

    print("Final performance: ")
    # logger.print_statistics()


if __name__ == "__main__":
    args = RGAT_parse_args()
    print(args)
    RGAT_main_procedure(args, dgl_model_flag=True)
