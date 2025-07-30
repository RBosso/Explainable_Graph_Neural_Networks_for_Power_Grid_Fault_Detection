

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import torch
import networkx as nx
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
print(os.getcwd())

# +
import time
#Extract Training Data

path = os.getcwd()

AdjacencyMatrix = np.load(path+"/AdjacencyMatrix.npy",allow_pickle = True)
print(AdjacencyMatrix)
print(AdjacencyMatrix.shape)
print(AdjacencyMatrix.size)

# #Define node values
# node_values = {}
# for n in range(len(BusesInOrder)):
#     node_values[n] = visualize_attr[n]

# print(node_values)


# print(AdjacencyMatrix)



# Create a graph from the adjacency matrix
G = nx.from_numpy_array(AdjacencyMatrix)


# Node location for visualization
pos = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (4, 0),
    5: (4, -0.25),
    6: (5, 0),
    7: (6, 0),
    8: (7, 0),
    9: (8, 0),
    10: (9, 0),
    11: (9, 0.25),
    12: (10, 0),
    13: (9, 0.50),
    14: (9, 0.75),
    15: (11, 0),
    16: (10, -1),
    17: (11, -1),
    18: (12, -1),
    19: (12, -0.20),
    20: (12, 0),
    21: (13.5, 0),
    22: (14.5, 0),
    23: (13.5, 0.2),
    24: (15.5, 0),
    25: (16.5, 0),
    26: (15.5, -0.25),
    27: (13.5, 0.4),
    28: (13.5, 0.6),
    29: (13.5, 0.8),
    30: (12, -0.4),
    31: (13.5, -0.20),
    32: (15, -1),
    33: (12, -0.6),
    34: (12, 0.2),
    35: (15.5, -0.5),
    36: (14.5, -0.20)
}

BusesInOrder = ['SRCE', '800', '802', '806', '808', '810', '812', '814', '814R', '850', '816', '818', 
        '824', '820', '822', '826', '828', '830', '854', '832', '858', '834',
 '860', '842', '836', '840', '862', '844', '846', '848', '852R', '888', '856', '852', '864', '838', '890']
node_labels = {}
for n in range(len(BusesInOrder)):
    node_labels[n] = BusesInOrder[n]
print(node_labels)

# # Set node color values
# nx.set_node_attributes(G, node_values, 'value')

# color_map = plt.cm.Oranges
# # color_map = plt.cm.Blues

# print(G.nodes)
# print(color_map)
# node_colors = [color_map(node_values[node]) for node in G.nodes]
# print(node_colors)

# Draw the graph
plt.figure(figsize=(7, 5))
#nx.draw(G, pos, with_labels=True,  node_size=655, labels = node_labels, edge_color='black', node_color='lightblue', node_shape='s',linewidths=1, font_size=11, vmin=0.0, vmax=1.0)

#nx.draw(G, pos,  node_size=85, edge_color='black', linewidths=1, width=1.5,font_size=12, vmin=0.0, vmax=1.0)

# Draw nodes with circle shape
nx.draw(G, pos, with_labels=False, node_size=15, edge_color='black', node_color='darkred', node_shape='s', linewidths=1, font_size=10)

# Draw labels separately to adjust their positions
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, horizontalalignment='left',verticalalignment='bottom')

# Display the plot
# plt.figure(figsize=(20, 18))
plt.show()


