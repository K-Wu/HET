#!/usr/bin/env python3
import networkx as nx


def generate_synthetic_graph(n_nodes, prob_edge, seed=None, rng_distribution="gnp"):
    # NB: we may alternatively use gnm_random_graph(n: int, m: int, seed: int = None, directed: bool = False) -> Graph
    # Other candidates are newman_watts_strogatz_graph(n, k, p, seed=None)
    # watts_strogatz_graph(n, k, p, seed=None)
    # For more info, visit https://networkx.org/documentation/stable/_modules/networkx/generators/random_graphs.html#fast_gnp_random_graph
    if rng_distribution == "gnp":
        g = nx.generators.random_graphs.fast_gnp_random_graph(
            n_nodes, prob_edge, seed=seed, directed=True
        )
    elif rng_distribution == "gnm":
        g = nx.generators.random_graphs.gnm_random_graph(
            n_nodes, int(n_nodes * prob_edge), seed=seed, directed=True
        )
    else:
        raise NotImplementedError("rng_distribution must be either gnp or gnm")
    return g


def generate_hetero_synthetic_graph():
    raise NotImplementedError
