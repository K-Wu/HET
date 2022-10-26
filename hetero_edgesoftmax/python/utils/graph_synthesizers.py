#!/usr/bin/env python3
import networkx as nx


def generate_synthetic_graph(n_nodes, prob_edge, seed=None):
    # NB: we may alternatively use gnm_random_graph whose API is identical to nx.fast_gnp_random_graph
    # Other candidates are newman_watts_strogatz_graph(n, k, p, seed=None)
    # watts_strogatz_graph(n, k, p, seed=None)
    # For more info, visit https://networkx.org/documentation/stable/_modules/networkx/generators/random_graphs.html#fast_gnp_random_graph
    g = nx.generators.random_graphs.fast_gnp_random_graph(
        n_nodes, prob_edge, seed=seed, directed=True
    )
    return g


def generate_hetero_synthetic_graph():
    raise NotImplementedError
