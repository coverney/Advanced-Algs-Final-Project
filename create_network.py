#!/usr/bin/env python3

"""
This script reads in the data for a network from its filenames
"""

import sys
import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.cm as cmx


def read_featnames(file):
    """
    Opens a .featnames file which has the feature names and converts it to a df.
    Used for the Twitter data
    """
    df = pd.read_csv(file, sep=" ", names=["featname"], index_col=0)
    return df


def read_feat(file):
    """
    Opens a .feat file which has the feature values for each node.
    The first elm in each line is the node name, all other values are binary.
    Used for the Twitter data
    """
    df = pd.read_csv(file, sep=" ", names=["node_id"] + list(range(0, 1364)))
    return df


def read_edge(file):
    """
    Reads a edge list and returns a networkX graph representing the edges
    """
    G = nx.read_edgelist(file, create_using=nx.DiGraph)
    return G


def visualize_graph(G, node_color, label, filename=None):
    """
    Visualize graph and color nodes by range of values in node_color and
    saves the image if a filename is given
    """
    node_color_min, node_color_max = min(node_color), max(node_color)
    viridis = cm = plt.get_cmap("viridis")
    cNorm = colors.Normalize(vmin=node_color_min, vmax=node_color_max)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=viridis)

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx(
        G, pos=pos, with_labels=False, node_color=node_color, node_size=150
    )
    ax.axis("off")
    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=0.05)
    sm = plt.cm.ScalarMappable(
        cmap=mpl.cm.viridis,
        norm=plt.Normalize(vmin=node_color_min, vmax=node_color_max),
    )
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.get_xaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_xlabel(label, rotation=0, size=13)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, format='png', bbox_inches='tight')

if __name__ == "__main__":
    # Test read_edge and visualize_graph for a twitter network
    if len(sys.argv) < 2:
        G = read_edge("Data/twitter/12831.edges")
    else:
        G = read_edge(sys.argv[1])
    print(nx.info(G))
    visualize_graph(G, [G.degree(v) for v in G], "Degree")
