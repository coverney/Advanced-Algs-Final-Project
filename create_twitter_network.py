'''
There are a total of 973 networks in the Twitter dataset
This script reads in the data for a network from its filenames

Notes:
- I am not sure if I am opening the .circles files correctly, because they
  don't seem to contain much info
- The .egofeat files also seem kind of useless

'''

import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def read_featnames(file):
    '''
    Opens a .featnames file which has the feature names and converts it to a df
    '''
    df = pd.read_csv(file, sep=' ', names=['featname'], index_col=0)
    return df

def read_feat(file):
    '''
    Opens a .feat file which has the feature values for each node.
    The first elm in each line is the node name, all other values are binary
    '''
    df = pd.read_csv(file, sep=' ', names=['node_id']+list(range(0, 1364)))
    return df

def read_edge(file):
    '''
    Reads a edge list and returns a networkX graph representing the edges
    '''
    G=nx.read_edgelist(file)
    return G

def visualize_graph(G):
    '''
    Visualize graph and color nodes by degree
    '''
    pos = nx.spring_layout(G)
    node_color = [2000.0 * G.degree(v) for v in G]
    min_color = min(node_color)/2000
    max_color = max(node_color)/2000
    fig,ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(G, pos=pos, with_labels=False,
                     node_color=node_color, node_size=150)
    ax.axis('off')
    # Create color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='3%', pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=mpl.cm.viridis, norm=plt.Normalize(vmin=min_color, vmax=max_color))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.ax.get_xaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_xlabel('Degree', rotation=0, size=13)
    plt.show()

if __name__ == "__main__":
    df_featnames = read_featnames('Data/twitter/12831.featnames')
    df_feat = read_feat('Data/twitter/12831.feat')
    G = read_edge('Data/twitter/12831.edges')
    print(nx.info(G))
    visualize_graph(G)
