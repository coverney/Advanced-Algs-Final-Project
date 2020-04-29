#!/usr/bin/env python3

'''
This script contains 3 different PageRank implementations
'''

import sys
from create_network import *
import numpy as np
import networkx as nx
import pandas as pd
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.preprocessing import normalize
import pickle
import os
from tqdm import tqdm


def preprocessing(G):
    '''
    Creates a google matrix from the adjacency matrix of an inputted
    networkX graph
    '''
    # Tunables
    alpha = 0.5

    H = nx.adjacency_matrix(G)
    # H = H.toarray()
    shape = H.shape
    assert shape[0] == shape[1]
    n = shape[0]

    # Replace dangling nodes with all 1's
    dangling_nodes =  np.where(H.getnnz(1)==0)[0] # Works for Scipy sparse arrays
    for i in dangling_nodes:
        # TODO: Use lil_matrix?
        H[i, :] = np.ones(n)  # Link to everyone, including yourself

    # assert not np.array([(1 if all(H[i, :] == 0) else 0) for i in range(n)]).any()
    assert not any(H.getnnz(1)==0)

    # Normalize
    H = H / H.sum(axis=1)
    assert not np.isnan(H).any()

    # constant_term = np.transpose((1-alpha)*np.ones(n))[np.newaxis]) * (np.ones(n)/n)
    constant_term = np.full_like(H, (1 - alpha) / n)
    variable_term = alpha * H
    google = constant_term + variable_term
    return google


def power_method(G, verbose=True):
    '''
    Runs the power method after calling preprocessing() and finding the dense
    google matrix
    '''
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
        prevpi = pi.copy()
        iters += 1

        pi = prevpi.dot(google)

        difference = np.linalg.norm(pi - prevpi)

    end = time.time()

    if verbose:
        print(f"Time elapsed: {end-start}")
        print(f"Number of iterations: {iters}")
        return np.asarray(pi)[0]
    else:
        return end-start, iters, np.asarray(pi)[0]


def numpy_method(G, verbose=True):
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

    if verbose:
        print(f"Time elapsed: {end-start}")
        return principal
    else:
        return end-start, principal

def sparse_power_method(G, verbose=True):
    '''
    Another way of using the power method to find the PageRank.
    Instead of pre-computing the google matrix, we calculate it using the
    sparse form of the adjacency matrix in each iteration of the power method.
    This allows us to save memory and time.
    '''
    start = time.time()
    # Tunables
    epsilon = 1e-8
    alpha = 0.5

    H = nx.adjacency_matrix(G)
    H = normalize(H, norm='l1', axis=1)
    shape = H.shape
    assert shape[0] == shape[1]
    n = shape[0]

    dangling_nodes = (H.getnnz(1)==0).astype(int)

    pi = 1 / n * np.ones(n)
    iters = 0
    difference = 1  # Arbitrary number larger than epsilon; it'll get overwritten

    while difference >= epsilon:
        prevpi = pi.copy()
        iters += 1
        # Do normal page-to-page transitions
        # This is the only step that involves the sparse matrix H
        pi = alpha * prevpi * H
        # Fix dangling nodes
        dangling_term = alpha * (prevpi * dangling_nodes)
        # Add in probability to teleport
        teleportation_constant = 1 - alpha # can be constant since will add to vector
        # Combine the likelihood of following connections and the likelihood
        # of teleporting for each elm, divide by n to normalize
        pi += (dangling_term + teleportation_constant) * (1/n)
        # Calculate difference between the two vectors
        difference = np.linalg.norm(pi - prevpi)

    end = time.time()

    if verbose:
        print(f"Time elapsed: {end-start}")
        print(f"Number of iterations: {iters}")
        return pi
    else:
        return end-start, iters, pi

def rank_nodes(pi, G, verbose=True):
    '''
    Ranks the nodes in a networkX graph by their ranks from PageRank
    '''
    nodes = list(G.nodes)
    assert len(nodes) == len(list(pi))
    pairs = zip(list(pi), nodes)
    # make sure nodes with the same rank have a predictable order
    order = sorted(pairs, key=lambda p: p[1])
    # sort pairs by rank (descending order)
    order = sorted(order, reverse=True)
    if verbose:
        print(order[:4])
    return order

def analyze_twitter(folder, n):
    '''
    Runs all 3 PageRank methods on the first n twitter networks, saving the
    graph visualizations and metrics.
    '''
    num_nodes = []
    method1_times = []
    method1_num_iters = []
    method2_times = []
    method3_times = []
    method3_num_iters = []
    directory = os.fsencode(folder)
    count = 0
    # go through the files in the twitter data folder
    for file in tqdm(os.listdir(directory)):
        if count > n:
            break
        file = file.decode("utf-8")
        # ignore the non edge files
        if not '.edges' in file:
            continue
        filename = 'Images/twitter'+file.split('.')[0]+'_ranking.png'
        # read the edge file as a networkX graph
        G = read_edge(folder+'/'+file)
        num_nodes.append(len(G.nodes))
        # run dense power method
        time1, iter1, pi1 = power_method(G, verbose=False)
        method1_times.append(time1)
        method1_num_iters.append(iter1)
        # run Numpy method
        time2, pi2 = numpy_method(G, verbose=False)
        method2_times.append(time2)
        # run sparse power method
        time3, iter3, pi3 = sparse_power_method(G, verbose=False)
        method3_times.append(time3)
        method3_num_iters.append(iter3)
        # visualize the graph
        visualize_graph(G, np.array(list(pi3)), 'PersonRank', filename)
        count += 1
    # save the metrics as a csv file
    df = pd.DataFrame({'num_node':num_nodes, 'method1_time':method1_times,
            'method1_num_iter':method1_num_iters, 'method2_time':method2_times,
            'method3_time':method3_times, 'method3_num_iter':method3_num_iters})
    df.to_csv('twitter_summary.csv', index=False)

def get_top_wikipages(file, n):
    '''
    Return the top n Wikipedia article names, sorted by ranking
    '''
    # read in the nodes ordered by ranking
    with open(file, 'rb') as filehandle:
        # read the data as binary data stream
        order = pickle.load(filehandle)
    print(order[:4])
    # read in the wikipedia article names
    df = pd.read_csv('Data/wiki-topcats-page-names.txt', names=["article_name"])
    # return the article names of the first 10 nodes in order
    names = []
    for rank, node in order[:n]:
        names.append(df.loc[int(node), 'article_name'])
    return names

if __name__ == "__main__":
    # Run the sparse power method on an edge file
    if len(sys.argv) < 2:
        filename = "Data/twitter/12831.edges"
        # filename = "Data/wiki-topcats.txt.gz"
    else:
        filename = sys.argv[1]
    G = read_edge(filename)
    print(len(G.nodes))
    pi = sparse_power_method(G)
    # prints out the first 4 nodes
    order = rank_nodes(pi, G)

    # # save the order as a pickle file
    # with open('ranking.data', 'wb') as filehandle:
    #     # store the data as binary data stream
    #     pickle.dump(order, filehandle)

    # # Analyze the first 50 twitter networks
    # analyze_twitter('Data/twitter', 50)

    # # Print the 10 highest rank Wikipedia articles
    # names = get_top_wikipages('wiki_ranking.data', 10)
    # print(names)
