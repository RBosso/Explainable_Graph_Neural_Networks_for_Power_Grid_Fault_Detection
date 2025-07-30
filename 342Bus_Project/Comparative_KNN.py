# %%

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

#
BusFile = "/342Bus/ieee390_EXP_VOLTAGES.CSV"

path = os.getcwd()
BusFile = path+BusFile
Buses = pd.read_csv(BusFile)
#Buses['Bus'].to_csv(path + '/Data/encoding_setup.csv',index = False)

Buses = Buses['Bus'].to_numpy()
print(Buses)
encoder = Buses.astype(str)
encoder = encoder.tolist()
print(encoder)
print(len(encoder))

research_paper_decoder = pd.read_csv(path + '/Data/encoding_setup.csv')
research_paper_decoder = research_paper_decoder['Encoding'].to_numpy()
research_paper_decoder = research_paper_decoder.tolist()
print(research_paper_decoder)
print(len(research_paper_decoder))
##
FaultLocationLabels = value_data[:,3]
print(FaultLocationLabels)
#print(np.unique(FaultLocationLabels))
#print(len(np.unique(FaultLocationLabels)))
for n in range(len(FaultLocationLabels)):
    FaultLocationLabels[n]=research_paper_decoder[encoder.index(str(FaultLocationLabels[n]))]
y = FaultLocationLabels.astype("int64")
print(y)
print(y.shape)
print(y.size)
#
X_train, X_test, y_train, y_test = train_test_split(numpy_data, y, test_size=0.20, random_state=41)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import torch
from torch.nn.functional import normalize
normalized_X_test = []
for n in range(len(X_test)):# change between numpy_data and silent_numpy_data

#    Uncomment for noise
#    noise = np.random.normal(1,0.03, size = (390,6)) #0.09
#    X_test[n] = X_test[n]*noise
    
    flattened_data = X_test[n]# change between numpy_data and silent_numpy_data
    flattened_data = flattened_data.astype("float32")
    flattened_data = torch.from_numpy(flattened_data)
    normalized_data = normalize(flattened_data)
    normalized_data = normalized_data.numpy()
    normalized_data = normalized_data.flatten()
 #   normalized_data = (flattened_data-flattened_data.mean())/flattened_data.std()
    normalized_X_test.append(normalized_data)

normalized_X_test = np.array(normalized_X_test)
print(X_test[0])
print("*"*88)
print(normalized_X_test[0])
print(normalized_X_test.shape)
print(normalized_X_test.size)
#
#
print("T"*88)
#
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

#
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, accuracy_score
#
model = KNeighborsClassifier(n_neighbors = 1)
KNeighborsClassifier = model.fit(normalized_X_train,y_train)
#print(y_test.shape)
print("Area Under ROC Curve for Naive Bayes =", roc_auc_score(y_test, KNeighborsClassifier.predict_proba(normalized_X_test), multi_class='ovr'))
#print(KNeighborsClassifier.predict_proba(normalized_X_test))
#print(KNeighborsClassifier.predict_proba(normalized_X_test).shape)
#

# print("Diagnose Area Under ROC Curve for kNN =")
# ROC_Per_class = roc_auc_score(y_test, KNeighborsClassifier.predict_proba(normalized_X_test), multi_class='ovr', average=None)

# for classes in range(len(ROC_Per_class)):
#     if ROC_Per_class[classes] < 0.7:
#         print(classes, ' has AUC of ',ROC_Per_class[classes])
        
#print(KNeighborsClassifier.predict(normalized_X_test))
#print(KNeighborsClassifier.predict(normalized_X_test).shape)

print("Training Accuracy =",accuracy_score(y_train, KNeighborsClassifier.predict(normalized_X_train)))
print("Testing Accuracy =",accuracy_score(y_test, KNeighborsClassifier.predict(normalized_X_test)))
#
from sklearn.metrics import f1_score
print("F1 Score =", f1_score(y_test, KNeighborsClassifier.predict(normalized_X_test), average = 'macro'))
##

###research_paper_decoder = [0,0,1,2,3,4,5,6,6,6,6,7,8,9,10,11,8,12,12,13,14,15,21,15,16,16,16,17,18,18,13,13,19,13,20,22,23]
###confusion_mat_labels = ["Src/800", "802","806","808","810","812","814/850/816","818","824/828","820","822","826","830/854","852/832/888","858",
###        "834/842","836/840/862","844","846/848","856","864","860","838","890"]
pred_list = KNeighborsClassifier.predict(normalized_X_test)
##y_test = y_test.astype(str)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred_list)#, labels=classes
print(cm)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 12)
plt.imshow(cm, interpolation = 'none', cmap = 'nipy_spectral_r')
# plt.imshow(cm, interpolation = 'none', cmap = 'gist_earth')
plt.colorbar()

# plt.title('Matrix Visualization')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.plot()
# import matplotlib.pyplot as plt 
# # plt.imshow(cm, interpolation = 'none') 
# plt.rcParams['figure.figsize'] = (12, 12)
# # plt.figure(figsize=(25, 25))
# plt.imshow(cm, interpolation = 'none', cmap = 'cividis') 

# plt.colorbar()

# # plt.title('Matrix Visualization')
# plt.xlabel('Predicted Label') 
# plt.ylabel('True Label') 

# #plt.plot() 
# plt.show() 


# %%
