#!/usr/bin/env python3

import sys
from create_twitter_network import *
import numpy as np
import networkx as nx
import pandas as pd
import time

def power_method(G):
    start = time.time()
    # Tunables
    alpha = 0.5
    epsilon = 1e-8

    H = nx.adjacency_matrix(G)
    # print(type(H))
    # print(H.get_shape())
    # H = np.asarray(H)  # H should be a sparse matrix
    # print(type(H))
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
    dangling_nodes = np.array([(1 if all(H[i, :] == 0) else 0) for i in range(n)])
    for i in dangling_nodes:
        H[i, :] = np.ones(n) # Link to everyone, including yourself

    # Normalize
    H = H / np.sum(H, axis=1)[:, np.newaxis]

    # constant_term = np.transpose((1-alpha)*np.ones(n))[np.newaxis]) * (np.ones(n)/n)
    constant_term = np.full_like(H, (1-alpha)/n)
    variable_term = alpha*H
    google = (constant_term + variable_term)

    # Make a starting vector
    # pi0 = np.random.random(H.shape[0]) # random noise start vector
    pi0 = 1/n*np.ones(n)[np.newaxis,:] # uniform start vector

    iters = 0
    difference = 1 # Arbitrary number larger than epsilon; it'll get overwritten
    pi = pi0

    while difference >= epsilon:
        prevpi = pi
        iters += 1

        pi = prevpi.dot(google)

        difference = np.linalg.norm(pi - prevpi)

    end = time.time()
    print(f"Time elapsed: {end-start}")
    print(f"Number of iterations: {iters}")
    return pi

def rank_nodes(pi, G):
    nodes = list(G.nodes)
    pairs = zip(nodes, list(pi))
    order = [x for _, x in sorted(pairs, key=lambda pair: -pair[1])]
    print(order[:3])

if __name__ == "__main__":
    if 1 <= len(sys.argv):
        filename = "Data/twitter/12831.featnames"
    else:
        filename = sys.argv[1]
    G = read_edge(filename)
    pi = power_method(G)
    rank_nodes(pi, G)
