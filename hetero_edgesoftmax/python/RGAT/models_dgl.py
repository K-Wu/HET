#!/usr/bin/env python3
# external code. @xiangsx knows the source.
"""RGAT layer implementation"""
import itertools
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
from .models import HET_RelationalAttLayer, HET_RelationalGATEncoder

# involve code heavily from dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph.

    Parameters
    ----------RelGraphEmbed
    g : DGLGraph
        Input graph.
    embed_size : int
        The length of each embedding vector
    exclude : list[str]
        The list of node-types to exclude (e.g., because they have natural features)
    """

    def __init__(self, g, embed_size, exclude=list()):

        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size

        # create learnable embeddings for all nodes, except those with a node-type in the "exclude" list
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if ntype in exclude:
                continue
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            self.embeds[ntype] = embed

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeds.values():
            nn.init.xavier_uniform_(emb)

    def forward(self, block=None):
        return self.embeds


class RelationalAttLayer(nn.Module):
    # corresponding to RelGraphConvLayer in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention layer.

    For inner relation message aggregation we use multi-head attention network.
    For cross relation message we just use average

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    n_heads : int
        Number of attention heads
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        n_heads,
        *,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0,
    ):
        super(RelationalAttLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(in_feat, out_feat // n_heads, n_heads, bias=False)
                for rel in rel_names
            }
        )  # NB: RGAT model definition

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))

        # self.reset_parameters()

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        if self.bias:
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )
        for module in self.conv.mods.values():
            module.reset_parameters()

    # pylint: disable=invalid-name
    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            h = h.view(-1, self.out_feat)
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        for k, _ in inputs.items():
            if g.number_of_dst_nodes(k) > 0:
                if k not in hs:
                    print(
                        "Warning. Graph convolution returned empty dictionary, "
                        f"for node with type: {str(k)}"
                    )
                    for _, in_v in inputs_src.items():
                        device = in_v.device
                    hs[k] = th.zeros(
                        (g.number_of_dst_nodes(k), self.out_feat), device=device
                    )
                    # TODO the above might fail if the device is a different GPU
                else:
                    hs[k] = hs[k].view(hs[k].shape[0], hs[k].shape[1] * hs[k].shape[2])

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RelationalGATEncoder(nn.Module):
    # corresponding to EntityClassify in dgl/examples/pytorch/ogb/ogbn-mag/hetero_rgcn.py
    r"""Relational graph attention encoder

    Parameters
    g : DGLHeteroGraph
        Input graph.
    h_dim: int
        Hidden dimension size
    out_dim: int
        Output dimension size
    n_heads: int
        Number of heads
    num_hidden_layers: int
        Num hidden layers
    dropout: float
        Dropout
    use_self_loop: bool
        Self loop
    last_layer_act: bool
        Whether add activation at the last layer
    """

    def __init__(
        self,
        g,
        h_dim,
        out_dim,
        n_heads,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=True,
        last_layer_act=False,
    ):
        super(RelationalGATEncoder, self).__init__()
        self.n_heads = n_heads
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.last_layer_act = last_layer_act
        self.init_encoder()

    def init_encoder(self):
        """Initialize RelationalGATEncoder encoder"""
        self.layers = nn.ModuleList()
        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                RelationalAttLayer(
                    self.h_dim,
                    self.h_dim,
                    self.g.etypes,
                    self.n_heads,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        # h2o
        self.layers.append(
            RelationalAttLayer(
                self.h_dim,
                self.out_dim,
                self.g.etypes,
                1,  # overwrting the n_head setting as the classification should be output in this stage
                activation=F.relu if self.last_layer_act else None,
                self_loop=self.use_self_loop,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, h=None, blocks=None):
        """Forward computation

        Parameters
        ----------
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        """
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h


def RGAT_parse_args():
    # DGL
    parser = argparse.ArgumentParser(description="RGAT")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout probability"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=64, help="number of hidden units"
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


def RGAT_get_model(g, num_classes, args):
    embed_layer = RelGraphEmbed(g, args["n_hidden"], exclude=[])  # exclude=["paper"])

    model = RelationalGATEncoder(
        g,
        h_dim=args["n_hidden"],
        out_dim=num_classes,
        n_heads=args["n_head"],
        num_hidden_layers=args["num_layers"] - 1,
        dropout=args["dropout"],
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
    embed_layer = RelGraphEmbed(g, args["n_hidden"], exclude=[])  # exclude=["paper"])

    model = HET_RelationalGATEncoder(
        g,
        h_dim=args["n_hidden"],
        out_dim=num_classes,
        n_heads=args["n_head"],
        num_hidden_layers=args["num_layers"] - 1,
        dropout=args["dropout"],
        use_self_loop=True,
    )

    print(embed_layer)
    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model


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
        num_layers=2,
        fanout=[25, 20],
        batch_size=1024,
    )
    hyperparameters.update(vars(args))
    print(hyperparameters)

    device = f"cuda:0" if th.cuda.is_available() else "cpu"

    (g, labels, num_classes, split_idx, train_loader) = prepare_data(hyperparameters)
    # TODO: now this script from dgl repo uses the num_classes properties of dataset. Align this with graphiler's randomizing label, or add an option whether to randomize classification.
    if dgl_model_flag:
        print("Using DGL RGAT model")
        embed_layer, model = RGAT_get_model(g, num_classes, hyperparameters)
    else:
        print("Using our RGAT model")
        embed_layer, model = RGAT_get_our_model(g, num_classes, hyperparameters)
    model = model.to(device)

    for run in range(hyperparameters["runs"]):
        embed_layer.reset_parameters()
        model.reset_parameters()

        # optimizer
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=hyperparameters["lr"])

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
    RGAT_main_procedure(args, dgl_model_flag=True)
