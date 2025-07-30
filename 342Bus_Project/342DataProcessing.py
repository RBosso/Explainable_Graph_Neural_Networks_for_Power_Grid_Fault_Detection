
import networkx as nx
import numpy as np
import pandas as pd
import time
import os
import sys

start = time.time()
#Path for linux
path = os.getcwd()
#non_noisy_data = pd.read_parquet(path + "/Data/342_AllFaultData.parquet")
#silent_columns = ["Bus", " Magnitude1", " Angle1", " Magnitude2", " Angle2", " Magnitude3", " Angle3","Fault_Command","PV_Irradiance","Fault_Type","FaultLocation"]
#non_noisy_data = non_noisy_data[silent_columns]
#
#print(os.getcwd())
#
#no_noise_numpy_data = non_noisy_data.to_numpy()
#print(no_noise_numpy_data)
#print(no_noise_numpy_data.shape)
#
#sectioner = []
#sectioner2 = []
#for n in range(len(no_noise_numpy_data)):
#    if n%390==0:
#        sectioner.append(n)
#print(sectioner[0],sectioner[1])
#print(len(sectioner))
#
#node_features = []
#node_labels = []
#for n in range(len(sectioner)):
#    if n+1 < len(sectioner):
#        ####print([sectioner[n],sectioner[n+1]])
#        node_features.append([no_noise_numpy_data[sectioner[n]:sectioner[n+1]][:,1:7]])
#        node_labels.append(no_noise_numpy_data[sectioner[n]:sectioner[n+1]][0,7:])
#    else: 
#        ####print([sectioner[n],])
#        node_features.append([no_noise_numpy_data[sectioner[n]:][:,1:7]])
#        node_labels.append(no_noise_numpy_data[sectioner[n]:][0,7:])
###print(len(node_features))
#
#np.save(path + "/new_data.npy",node_features)
#np.save(path + "/new_labels.npy",node_labels)
#print("This took ",-start+end, "seconds")


numpy_data = np.load(path+"/new_data.npy",allow_pickle = True)
numpy_data = np.squeeze(numpy_data)
print(numpy_data)
print(numpy_data.shape)
print(numpy_data.size)

value_data = np.load(path+"/new_labels.npy",allow_pickle = True)
print(value_data)
print(value_data.shape)
print(value_data.size)

end = time.time()
print("This took ",-start+end, "seconds")
# Below processes Adjacency
BusFile = "/342Bus/ieee390_EXP_VOLTAGES.CSV"

path = os.getcwd()
BusFile = path+BusFile
Buses = pd.read_csv(BusFile)
#Buses = Buses[~Buses['Bus'].str.contains('_OPEN')] 
Buses = Buses['Bus'].to_numpy()
print(Buses)

lines = pd.read_csv(path+"/342Bus/Source/Lines.csv", skiprows=1)
print(lines)
print(lines.columns)
line_connections = lines[[' Bus1',' Bus2']]
line_connections = line_connections.to_numpy().tolist() 
print(line_connections)

X_axis = Buses
Y_axis = Buses

AdjacencyMatrix = np.zeros((len(X_axis),len(X_axis)))
#print(X_axis)
# #print(AdjacencyMatrix)
X_count = 0
Y_count = 0

for X in X_axis:
    Y_count = 0
    for Y in Y_axis:
        if ([X,Y] in line_connections) and X != Y:
            #print(X,Y)
            AdjacencyMatrix[Y_count][X_count] = 1
            AdjacencyMatrix[X_count][Y_count] = 1
        Y_count +=1
    X_count+=1
#
#np.set_printoptions(threshold=sys.maxsize)
print(AdjacencyMatrix)
print(AdjacencyMatrix.shape)

 # #print(AdjacencyMatrix == AdjacencyMatrix.T)
if (False not in (AdjacencyMatrix == AdjacencyMatrix.T)):
    print("Adjacency Matrix is Symmetrical")
else:
    print("Not Symmetrical")

np.save(path + "/AdjacencyMatrix.npy",AdjacencyMatrix)

#For encoding the node indices in a paired array:
encoder = Buses.astype(str)
encoder = encoder.tolist()
print(encoder)
print()
# Encodes buses by location in node feature matrix 
for pair in range(len(line_connections)):
    #    print(line_connections[pair][0])
#    print(lines_list[pair][1])    #print(pair))
    line_connections[pair][0] = encoder.index(line_connections[pair][0])
    line_connections[pair][1] = encoder.index(line_connections[pair][1])
print(line_connections)

# Removes self-referencing connections
removal = []
for pair in range(len(line_connections)):
    if line_connections[pair][0] == line_connections[pair][1]:
        removal.append(pair)
for r in reversed(removal):
    line_connections.pop(r)

# Adds reversed node connection directions to make adjacency bidirectional and symmetrical
reversed = []
for pair in line_connections:
    reversed.append([pair[1],pair[0]])
line_connections = line_connections + reversed

#Removes Duplicates from line_connections list
no_duplicates = []
for data in line_connections:
    if data not in no_duplicates:
        no_duplicates.append(data)
print(no_duplicates)
print(len(no_duplicates))
print("T"*400)

np.save(path + "/NodeIndex.npy",no_duplicates)
#print(graphdata)

NodeIndex = np.load(path+"/NodeIndex.npy",allow_pickle = True)
print(NodeIndex)
print(NodeIndex.shape)
print(NodeIndex.size)

AdjacencyMatrix = np.load(path+"/AdjacencyMatrix.npy",allow_pickle = True)
print(AdjacencyMatrix)
print(AdjacencyMatrix.shape)
print(AdjacencyMatrix.size)
