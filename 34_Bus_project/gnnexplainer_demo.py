#from torch_geometric.data import Data, DataLoader
#from torch_geometric.datasets import TUDataset, Planetoid
#from torch_geometric.nn import GCNConv, Set2Set
#from torch_geometric.explain import GNNExplainer
#import torch_geometric.transforms as T
#import torch
#import torch.nn.functional as F
#import os
#from tqdm import tqdm, trange
#
#import matplotlib.pyplot as plt

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

numpy_data = np.load(path+"/new_data.npy",allow_pickle = True)
numpy_data = np.squeeze(numpy_data)
print(numpy_data)
print(numpy_data.shape)
print(numpy_data.size)

value_data = np.load(path+"/new_labels.npy",allow_pickle = True)
print(value_data)
print(value_data.shape)
print(value_data.size)


NodeIndex = np.load(path+"/NodeIndex.npy",allow_pickle = True)
print(NodeIndex)
print(NodeIndex.shape)
print(NodeIndex.size)

AdjacencyMatrix = np.load(path+"/AdjacencyMatrix.npy",allow_pickle = True)
print(AdjacencyMatrix)
print(AdjacencyMatrix.shape)
print(AdjacencyMatrix.size)

# +
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch.nn.functional import normalize

encoder = ['SourceBus', '800', '802', '806', '808', '810', '812', '814', '814r', '850',
'816', '818', '824', '820', '822', '826', '828', '830', '854', '832', '858',
'834', '860', '842', '836', '840', '862', '844', '846', '848', '852r', '888', '856', '852', '864', '838', '890']
print(encoder)
print(len(encoder))
research_paper_decoder = [0,0,1,2,3,4,5,6,6,6,6,7,8,9,10,11,8,12,12,13,14,15,21,15,16,16,16,17,18,18,13,13,19,13,20,22,23]

FaultLocationLabels = value_data[:,3]

for n in range(len(FaultLocationLabels)):
    FaultLocationLabels[n]=research_paper_decoder[encoder.index(str(FaultLocationLabels[n]))]

y = FaultLocationLabels.astype("int64")
y = torch.from_numpy(y)
x = numpy_data.astype("float32")
x = torch.from_numpy(x)

NodeIndex = NodeIndex.astype("int64")
NodeIndex = NodeIndex.T
NodeIndex = torch.from_numpy(NodeIndex)

print(y)
print(y.dtype)
print(x)
print(x.dtype)
print(NodeIndex)
print(NodeIndex.dtype)
#
print(x[0])
print(normalize(x[0]))
#
result_translator = np.unique(FaultLocationLabels.astype("int64")).tolist()
print()
total_data_list = []
for n in range(len(x)):
    #print(x[n])
    #print(y[n])
    DataObject = Data(x = x[n], edge_index = NodeIndex, y = y[n], is_undirected = True) #Testing with non-normalized data
    #DataObject = Data(x = x[n], edge_index = NodeIndex, y = y[n], is_undirected = True)
    DataObject.is_undirected = True
    total_data_list.append(DataObject)
#print('Y'*888)
#print(total_data_list[0].x)

print()
#print(f'Dataset: {total_data_list}:')
print('===================')
print(f'Number of graphs: {len(total_data_list)}')
print(f'Number of features: {total_data_list[0].num_features}')
#print(f'Number of classes: {total_data_list[0].num_classes}')

data = total_data_list[0]  # Get the first graph object.
#print(data)
#print(data.y)


#################################################################################
#import torch
#import torch.nn.functional as F
#from sklearn.metrics import roc_auc_score
#from sklearn.model_selection import train_test_split
#from tqdm import tqdm
#
#import torch_geometric.transforms as T
#from torch_geometric.datasets import ExplainerDataset
#from torch_geometric.datasets.graph_generator import BAGraph
#from torch_geometric.explain import Explainer, GNNExplainer
#from torch_geometric.nn import GCN
#from torch_geometric.utils import k_hop_subgraph
#
##dataset = ExplainerDataset(
##    graph_generator=BAGraph(num_nodes=300, num_edges=5),
##    motif_generator='house',
##    num_motifs=80,
##    transform=T.Constant(),
##)
#data = dataset[0]
#print(data)
