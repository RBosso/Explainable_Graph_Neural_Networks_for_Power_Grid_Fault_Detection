# Implementation and Libraries for GradCAM
from typing import Union, Tuple, Any
import os
import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_additional_forward_args,
    _format_tensor_into_tuples,
    _format_output,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType
from captum.attr import LayerGradCam
from torch import Tensor

class GraphLayerGradCam(LayerGradCam):
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        undo_gradient_requirements(inputs, gradient_mask)

        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=0,
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)
        return _format_output(len(scaled_acts) > 1, scaled_acts)

#Loading saved model architecture
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.pool import global_max_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(6, 512, improved = True)
        # self.conv2 = GCNConv(512, 256, improved = True)

        # self.conv1 = ChebConv(6, 512, K = 2)
        # self.conv2 = ChebConv(512,256, K = 2)

        self.conv1 = ChebConv(6, 256, K = 3)
        self.conv2 = ChebConv(256,256, K = 4)
        self.conv3 = ChebConv(256, 256, K = 5)
        self.fc1 = Linear(256, 512)
        self.fc2 = Linear(512, 256)

        # self.conv1 = GATConv(6, 256, heads = 4)
        # self.conv2 = GATConv(256*4,64, heads = 1, concat=False)

        # self.conv1 = TransformerConv(6, 200, heads = 3)
        # self.conv2 = TransformerConv(200*3,256, heads = 1, concat=False)

        self.lin = Linear(256, 24)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        # = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.2, training=self.training) #0.5
        x = self.lin(x)

        return x

model = GCN(hidden_channels=464)
print(model)

print(os.getcwd()+ "/model.pth")
model_save_location = os.getcwd()+ "/model.pth"       
model.load_state_dict(torch.load(model_save_location))
print(model.conv3)
print(model.state_dict())

#Loads IEEE34 Bus Simulation Data

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

print()
print(NormalizeFeatures(data.x))
print(data.x)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

from random import shuffle
torch.manual_seed(12345)
#total_data_list = total_data_list.shuffle()
shuffle(total_data_list)

train_dataset = total_data_list[:14640] #9150 is half
test_dataset = total_data_list[14640:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


print(test_dataset[0].x)
print(test_dataset[0])
#print(train_dataset[0].x)

for sample in range(len(test_dataset)):
    # noise = np.random.normal(1,0.03, size = (37,6)) #0.09 #Uncomment for noise
    # noise = noise.astype("float32")
    # noise = torch.from_numpy(noise)
    # test_dataset[sample].x = noise*test_dataset[sample].x
    test_dataset[sample].x = normalize(test_dataset[sample].x)
    #print(sample)

print('Y'*888)
#print(train_dataset[0].x)
print(test_dataset[0].x)
print(test_dataset[0])
##noise = np.random.normal(1,0.09, size = (16,6))
##noise = noise.astype("float32")
##noise = torch.from_numpy(noise)
print('=============================================================')
##
print(train_dataset[0].x)
print(train_dataset[0])
#print(train_dataset[0].x)

for sample in range(len(train_dataset)):
#    noise = np.random.normal(1,0.09, size = (16,6))
#    #    noise = np.random.normal(1,0.09, size = (1,96))
#    noise = noise.astype("float32")
#    noise = torch.from_numpy(noise)
##    print(noise)
##    X_test[sample] = X_test[sample]*noise[0]
#    train_dataset[sample].x = noise*train_dataset[sample].x
    train_dataset[sample].x = normalize(train_dataset[sample].x)
    #print(sample)

print('Y'*888)
#print(train_dataset[0].x)
print(train_dataset[0].x)
print(train_dataset[0])
##print(noise*test_dataset[0].x)
# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("$"*200)

# Example based on Captum Documentation
layer_gc = LayerGradCam(model, model.conv3)#

testgrad = torch.randn(3,3)
print(testgrad)
print(test_loader)
count = 0
for batch in test_loader:
   count+=1
#   print(count)
   print(batch)
   print(batch.batch)
   print(batch.edge_index)
   Edge_Index = batch.edge_index[:,:72]
   Batch = batch.batch[:37]
   print(batch.y)
   print(batch.x)
   print(batch.y[0])
   print(batch.x[0:37])
   label1 = batch.y[0].numpy().tolist()#
   input = batch.x[0:37]#
#   print(batch.x[37:74])
#   print(batch.x[74:111])
   break
print(label1)
print(input)
print(Edge_Index)
print(Batch)
print(label1.shape)
print(input.shape)
#print(NodeIndex)
#print(NodeIndex.shape)
#print(Edge_Index == NodeIndex)
print(Edge_Index.shape)
print(Batch.shape)
print(Batch[:37])
print(Batch[:37].shape)

attr = layer_gc.attribute(input, target = label1, additional_forward_args=(Edge_Index,Batch))
print(attr)
