

import networkx as nx
import numpy as np
import pandas as pd
import time
import os
import sys

start = time.time()
#Path for linux
path = os.getcwd()
non_noisy_data = pd.read_csv(path + "/Data/newest_AllFaultData.csv")
silent_columns = ["Bus", " Magnitude1", " Angle1", " Magnitude2", " Angle2", " Magnitude3", " Angle3","Fault_Command","PV_Irradiance","Fault_Type","FaultLocation"]
non_noisy_data = non_noisy_data[silent_columns]

print(os.getcwd())

no_noise_numpy_data = non_noisy_data.to_numpy()
print(no_noise_numpy_data)
print(no_noise_numpy_data.shape)

sectioner = []
sectioner2 = []
for n in range(len(no_noise_numpy_data)):
    if n%37==0:
        sectioner.append(n)
print(sectioner[0],sectioner[1])
print(len(sectioner))

node_features = []
node_labels = []
for n in range(len(sectioner)):
    if n+1 < len(sectioner):
        ####print([sectioner[n],sectioner[n+1]])
        node_features.append([no_noise_numpy_data[sectioner[n]:sectioner[n+1]][:,1:7]])
        node_labels.append(no_noise_numpy_data[sectioner[n]:sectioner[n+1]][0,7:])
    else: 
        ####print([sectioner[n],])
        node_features.append([no_noise_numpy_data[sectioner[n]:][:,1:7]])
        node_labels.append(no_noise_numpy_data[sectioner[n]:][0,7:])
###print(len(node_features))

np.save(path + "/new_data.npy",node_features)
np.save(path + "/new_labels.npy",node_labels)
end = time.time()
print("This took ",-start+end, "seconds")


numpy_data = np.load(path+"/new_data.npy",allow_pickle = True)
numpy_data = np.squeeze(numpy_data)
print(numpy_data)
print(numpy_data.shape)
print(numpy_data.size)

value_data = np.load(path+"/new_labels.npy",allow_pickle = True)
print(value_data)
print(value_data.shape)
print(value_data.size)

# Below processes Adjacency
BusFile = "/34Bus/ieee34-1_EXP_VOLTAGES.CSV"
#Note to Future Self: Just change the root csv, add a header such that SourceBus is included

path = os.getcwd()
BusFile = path+BusFile
Buses = pd.read_csv(BusFile)
Buses = Buses['Bus'].to_numpy()
print(Buses)

CircuitTopology = """Element             Bus1     Bus2     Bus3     ...

"Transformer.SUBXF" SOURCEBUS 800       
"Line.L1"           800       802       
"Line.L2"           802       806       
"Line.L3"           806       808       
"Line.L4"           808       810       
"Line.L5"           808       812       
"Line.L6"           812       814       
"Line.L7"           814R      850       
"Line.L8"           816       818       
"Line.L9"           816       824       
"Line.L10"          818       820       
"Line.L11"          820       822       
"Line.L12"          824       826       
"Line.L13"          824       828       
"Line.L14"          828       830       
"Line.L15"          830       854       
"Line.L16"          832       858       
"Line.L17"          834       860       
"Line.L18"          834       842       
"Line.L19"          836       840       
"Line.L20"          836       862       
"Line.L21"          842       844       
"Line.L22"          844       846       
"Line.L23"          846       848       
"Line.L24"          850       816       
"Line.L25"          852R      832       
"Transformer.XFM1"  832       888       
"Line.L26"          854       856       
"Line.L27"          854       852       
"Line.L28"          858       864       
"Line.L29"          858       834       
"Line.L30"          860       836       
"Line.L31"          862       838       
"Line.L32"          888       890       
"Capacitor.C844"    844       844       
"Capacitor.C848"    848       848       
"Transformer.REG1A" 814       814R      
"Transformer.REG1B" 814       814R      
"Transformer.REG1C" 814       814R      
"Transformer.REG2A" 852       852R      
"Transformer.REG2B" 852       852R      
"Transformer.REG2C" 852       852R"""      

formatTopo = CircuitTopology.splitlines()
print(formatTopo)

graphdata = []
for rows in formatTopo:
    # #print(rows.split()[0:3])
    if len(rows) >0:
        graphdata.append(rows.split()[1:3])
    # #print(rows.split())

graphdataframe = pd.DataFrame(graphdata[1:], columns=graphdata[0])
print(graphdataframe)
print(len(graphdataframe))

X_axis = Buses
Y_axis = Buses

AdjacencyMatrix = np.zeros((len(X_axis),len(X_axis)))
#print(X_axis)
# #print(AdjacencyMatrix)
X_count = 0
Y_count = 0
#
for X in X_axis:
    Y_count = 0
    for Y in Y_axis:
        # #print(X,Y)
        # #print(X_count, Y_count)
        # Y_count +=1
        # #print(X,Y)
        if ([X,Y] in graphdata) and X != Y:
            #print(X,Y)
            AdjacencyMatrix[Y_count][X_count] = 1
            AdjacencyMatrix[X_count][Y_count] = 1
        Y_count +=1
    X_count+=1

# #print(['Bus1', 'Bus2'] in graphdata)

#np.set_printoptions(threshold=sys.maxsize)
print(AdjacencyMatrix)

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
graphdata = graphdata[1:]

for pair in range(len(graphdata)):
    #print(pair)
    graphdata[pair][0] = encoder.index(graphdata[pair][0])
    graphdata[pair][1] = encoder.index(graphdata[pair][1])
removal = []
for pair in range(len(graphdata)):
    if graphdata[pair][0] == graphdata[pair][1]:
        removal.append(pair)
for r in reversed(removal):
    graphdata.pop(r)
reversed = []
for pair in graphdata:
    reversed.append([pair[1],pair[0]])
graphdata = graphdata + reversed

#Removes Duplicates from graphdata list
no_duplicates = []
for data in graphdata:
    if data not in no_duplicates:
        no_duplicates.append(data)
print(no_duplicates)
print(len(no_duplicates))
#print("T"*400)

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
print(len(graphdata))
