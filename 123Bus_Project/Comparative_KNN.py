# %%

# %%

# %% [markdown]
# Comparative Classification With KNN
# %%

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sklearn 
print(os.getcwd())
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier                            
# %%
from sklearn.model_selection import train_test_split
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


BusFile = "/123Bus/ieee123_EXP_VOLTAGES.CSV"

path = os.getcwd()
BusFile = path+BusFile
Buses = pd.read_csv(BusFile)
Buses = Buses[~Buses['Bus'].str.contains('_OPEN')] 
Buses = Buses['Bus'].to_numpy()
print(Buses)
encoder = Buses.astype(str)
encoder = encoder.tolist()
print(encoder)
print(len(encoder))
#encoder = ['SourceBus', '800', '802', '806', '808', '810', '812', '814', '814r', '850', 
#'816', '818', '824', '820', '822', '826', '828', '830', '854', '832', '858', 
#'834', '860', '842', '836', '840', '862', '844', '846', '848', '852r', '888', '856', '852', '864', '838', '890']
#research_paper_decoder = [0,0,1,2,3,4,5,6,6,6,6,7,8,9,10,11,8,12,12,13,14,15,21,15,16,16,16,17,18,18,13,13,19,13,20,22,23]

#['150', '150R', '149', '1', '2', '3', '7', '4', '5', '6', '8', '12',
#'9', '13', '9R', '14', '34', '18', '11', '10', '15', '16', '17', '19',
#'21', '20', '22', '23', '24', '25', '25R', '26', '28', '27', '31', 
#'33', '29', '30', '250', '32', '35', '36', '40', '37', '38', '39', 
#'41', '42', '43', '44', '45', '47', '46', '48', '49', '50', '51', 
#'151', '52', '53', '54', '55', '57', '56', '58', '60', '59', '61', 
#'62', '63', '64', '65', '66', '67', '68', '72', '97', '69', '70', 
#'71', '73', '76', '74', '75', '77', '86', '78', '79', '80', '81', 
#'82', '84', '83', '85', '87', '88', '89', '90', '91', '92', '93', 
#'94', '95', '96', '98', '99', '100', '450', '197', '101', '102', 
#'105', '103', '104', '106', '108', '107', '109', '300', '110', 
#'111', '112', '113', '114', '135', '152', '160R', '160', '61S', '610']
research_paper_decoder = [0,0,0,1,2,3,4,5,6,7,8,9,10,11,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,72,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,14,11,61,61,63,120]
print(research_paper_decoder)
print(len(research_paper_decoder))
#
#for n in range(len(Buses)):
#    print(research_paper_decoder[n], Buses[n])
FaultLocationLabels = value_data[:,3]
print(FaultLocationLabels)
for n in range(len(FaultLocationLabels)):
    FaultLocationLabels[n]=research_paper_decoder[encoder.index(str(FaultLocationLabels[n]).upper())]
y = FaultLocationLabels.astype("int64")
print(y)
print(y.shape)
print(y.size)

X_train, X_test, y_train, y_test = train_test_split(numpy_data, y, test_size=0.50, random_state=46)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import torch
from torch.nn.functional import normalize
normalized_X_test = []
for n in range(len(X_test)):# change between numpy_data and silent_numpy_data
    noise = np.random.normal(1,0.03, size = (130,6)) #0.09
    X_test[n] = X_test[n]*noise
    
#    std = X_test.std()
    #"A random Gaussian noise with 3% standard deviation is added to the input data" -quote from Nevada Paper
    #"A random noise of size 0-3% of original power flow results are generated and inserted into measurement data" -quote from Noise Paper
#    noise = np.random.normal(0,0.03, size = (16,6)) #0.09
#    X_test[n] = X_test[n]+X_test[n]*noise
    
    flattened_data = X_test[n]# change between numpy_data and silent_numpy_data
    flattened_data = flattened_data.astype("float32")
    flattened_data = torch.from_numpy(flattened_data)
    normalized_data = normalize(flattened_data)
    normalized_data = normalized_data.numpy()
    normalized_data = normalized_data.flatten()
#    normalized_data = (flattened_data-flattened_data.mean())/flattened_data.std()
    normalized_X_test.append(normalized_data)

normalized_X_test = np.array(normalized_X_test)
print(X_test[0])
print("*"*88)
print(normalized_X_test[0])
print(normalized_X_test.shape)
print(normalized_X_test.size)


print("T"*88)

normalized_X_train = []
for n in range(len(X_train)):# change between numpy_data and silent_numpy_data
    flattened_data = X_train[n]# change between numpy_data and silent_numpy_data
    flattened_data = flattened_data.astype("float32")
    flattened_data = torch.from_numpy(flattened_data)
    normalized_data = normalize(flattened_data)
    normalized_data = normalized_data.numpy()
    normalized_data = normalized_data.flatten()
#    normalized_data = (flattened_data-flattened_data.mean())/flattened_data.std()
    normalized_X_train.append(normalized_data)

normalized_X_train = np.array(normalized_X_train)
print(X_train[0])
print("*"*88)
print(normalized_X_train[0])
print(normalized_X_train.shape)
print(normalized_X_train.size)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, accuracy_score

model = KNeighborsClassifier(n_neighbors = 1)
KNeighborsClassifier = model.fit(normalized_X_train,y_train)
print("Area Under ROC Curve for Naive Bayes =", roc_auc_score(y_test, KNeighborsClassifier.predict_proba(normalized_X_test), multi_class='ovr'))
print(KNeighborsClassifier.predict_proba(normalized_X_test))
print(KNeighborsClassifier.predict_proba(normalized_X_test).shape)
#
#
print(KNeighborsClassifier.predict(normalized_X_test))
print(KNeighborsClassifier.predict(normalized_X_test).shape)

print("Training Accuracy =",accuracy_score(y_train, KNeighborsClassifier.predict(normalized_X_train)))
print("Testing Accuracy =",accuracy_score(y_test, KNeighborsClassifier.predict(normalized_X_test)))

#print(encoder) 

from sklearn.metrics import f1_score
print("F1 Score =", f1_score(y_test, KNeighborsClassifier.predict(normalized_X_test), average = 'macro'))
#
#research_paper_decoder = [0,0,1,2,3,4,5,6,6,6,6,7,8,9,10,11,8,12,12,13,14,15,21,15,16,16,16,17,18,18,13,13,19,13,20,22,23]
#confusion_mat_labels = ["Src/800", "802","806","808","810","812","814/850/816","818","824/828","820","822","826","830/854","852/832/888","858",
#        "834/842","836/840/862","844","846/848","856","864","860","838","890"]
pred_list = KNeighborsClassifier.predict(normalized_X_test)
y_test = y_test.astype(str)
pred_list = pred_list.astype(str)
#
#for faults in range(24):
#    pred_list[pred_list==faults] = confusion_mat_labels[faults]    
#    y_test[y_test==faults] = confusion_mat_labels[faults]    
# %%

##plot_confusion_matrix(clf, X_test, y_test, ax=ax)
##xtick_labels = np.arange(len(confusion_mat_labels))
cm = confusion_matrix(y_test, pred_list)#, labels=classes
#cm = confusion_matrix(y_test, pred_list, labels=confusion_mat_labels)#, labels=classes
cmp = ConfusionMatrixDisplay(cm)#.plot(xticks_rotation = 35)#.ax_.set_title("kNN Confusion Matrix")
#cmp = ConfusionMatrixDisplay(cm, display_labels=confusion_mat_labels)#.plot(xticks_rotation = 35)#.ax_.set_title("kNN Confusion Matrix")
fig, ax = plt.subplots(figsize=(22, 22)) 
##ax.set_xticks(xtick_labels-12)
cmp.plot(ax=ax, xticks_rotation = 40)
plt.xticks(np.arange(len(np.unique(y_test))), np.unique(y_test), ha='right')
plt.show()
