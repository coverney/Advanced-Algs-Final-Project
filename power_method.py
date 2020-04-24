#!/usr/bin/env python3

import sys
from create_twitter_network import *
import numpy as np
import networkx as nx
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.cm as cmx


def preprocessing(G):
    # Tunables
    alpha = 0.5

    H = nx.adjacency_matrix(G)
    H = H.toarray()
    shape = H.shape
    assert shape[0] == shape[1]
    n = shape[0]

    """
    # Row-normalize the matrix
    sums = np.sum(H, axis=1)
    H = np.transpose(np.divide(np.transpose(H), sums, out=np.zeros_like(H), where = sums!=0))
    assert H.shape == (n, n)
    """

    # Replace dangling nodes with all 1's
    dangling_nodes = np.where(~H.any(axis=1))[0]
    # print(dangling_nodes)
    for i in dangling_nodes:
        H[i, :] = np.ones(n)  # Link to everyone, including yourself

    assert not np.array([(1 if all(H[i, :] == 0) else 0) for i in range(n)]).any()

    # Normalize
    H = H / np.sum(H, axis=1)[:, np.newaxis]
    assert not np.isnan(H).any()

    # constant_term = np.transpose((1-alpha)*np.ones(n))[np.newaxis]) * (np.ones(n)/n)
    constant_term = np.full_like(H, (1 - alpha) / n)
    variable_term = alpha * H
    google = constant_term + variable_term
    return google


def power_method(G):
    start = time.time()

    # Tunables
    epsilon = 1e-8

    google = preprocessing(G)
    n = google.shape[0]
    # Make a starting vector
    # pi0 = np.random.random(H.shape[0]) # random noise start vector
    pi0 = 1 / n * np.ones(n)[np.newaxis, :]  # uniform start vector

    iters = 0
    difference = 1  # Arbitrary number larger than epsilon; it'll get overwritten
    pi = pi0

    while difference >= epsilon:
        prevpi = pi
        iters += 1

        pi = prevpi.dot(google)

        difference = np.linalg.norm(pi - prevpi)

    end = time.time()
    print(f"Time elapsed: {end-start}")
    print(f"Number of iterations: {iters}")
    return pi[0]


def numpy_method(G):
    """
    Using Numpy's built-in eigenvalue function, find PageRank
    """

    # Get eigenvalues and eigenvectors.  The vectors are columns, such that
    # vecs[:,i] is the i'th eigenvector.
    start = time.time()
    google = preprocessing(G)

    # We need left eigenvectors, so we transpose the google matrix
    vals, vecs = np.linalg.eig(np.transpose(google))

    principal = vecs[:, np.argmax(np.abs(vals))]
    principal /= np.sum(principal)  # Normalize it
    assert np.abs(np.sum(principal) - 1) < 0.0001  # Roughly 1

    end = time.time()
    print(f"Time elapsed: {end-start}")
    return principal


def rank_nodes(pi, G):
    nodes = list(G.nodes)
    assert len(nodes) == len(list(pi))
    pairs = zip(list(pi), nodes)
    order = sorted(pairs, key=lambda p: p[1])
    order = sorted(order, reverse=True)
    print(order[:4])
    return order


if __name__ == "__main__":
    if len(sys.argv) < 2:
        filename = "Data/twitter/12831.edges"
    else:
        filename = sys.argv[1]
    G = read_edge(filename)
    # print(G.in_edges('180505807'))
    # print(G.out_edges('180505807'))
    pi1 = power_method(G)
    order1 = rank_nodes(pi1, G)
    pi2 = numpy_method(G)
    order2 = rank_nodes(pi2, G)
    # visualize_graph(G, np.array([x for x,_ in order]), 'PersonRank')
