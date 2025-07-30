# %%

# %% [markdown]
# Comparative Classification With Sklearn Models
# %%

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sklearn 
print(os.getcwd())
from sklearn.naive_bayes import GaussianNB
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

BusFile = "/342Bus/ieee390_EXP_VOLTAGES.CSV"

path = os.getcwd()
BusFile = path+BusFile
Buses = pd.read_csv(BusFile)
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
for n in range(len(FaultLocationLabels)):
    FaultLocationLabels[n]=research_paper_decoder[encoder.index(str(FaultLocationLabels[n]))]
y = FaultLocationLabels.astype("int64")
print(y)
print(y.shape)
print(y.size)

# %%

#
X_train, X_test, y_train, y_test = train_test_split(numpy_data, y, test_size=0.20, random_state=46)
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
#    normalized_data = (flattened_data-flattened_data.mean())/flattened_data.std()
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

model = GaussianNB()
GaussClassifier = model.fit(normalized_X_train,y_train)
print("Area Under ROC Curve for Naive Bayes =", roc_auc_score(y_test, GaussClassifier.predict_proba(normalized_X_test), multi_class='ovr'))
#print(GaussClassifier.predict_proba(normalized_X_test))
#print(GaussClassifier.predict_proba(normalized_X_test).shape)
#print(GaussClassifier.predict(normalized_X_test))
#print(GaussClassifier.predict(normalized_X_test).shape)

print("Training Accuracy =",accuracy_score(y_train, GaussClassifier.predict(normalized_X_train)))
print("Testing Accuracy =",accuracy_score(y_test, GaussClassifier.predict(normalized_X_test)))
print(encoder) 

from sklearn.metrics import f1_score
print("F1 Score =", f1_score(y_test, GaussClassifier.predict(normalized_X_test), average = 'macro'))

pred_list = GaussClassifier.predict(normalized_X_test)

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


# %%
#MLP Implementation for Model Comparison
X_train, X_test, y_train, y_test = train_test_split(numpy_data, y, test_size=0.20, random_state=78)
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
#
normalized_X_train = np.array(normalized_X_train)
print(X_train[0])
print("*"*88)
print(normalized_X_train[0])
print(normalized_X_train.shape)
print(normalized_X_train.size)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier                              

model = MLPClassifier(hidden_layer_sizes= (100,), random_state = 62, max_iter = 200).fit(normalized_X_train, y_train)

print("Area Under ROC Curve for MLP =", roc_auc_score(y_test, model.predict_proba(normalized_X_test), multi_class='ovr'))
print(model.predict_proba(normalized_X_test))
print(model.predict_proba(normalized_X_test).shape)

print("Training Accuracy =",accuracy_score(y_train, model.predict(normalized_X_train)))
print("Testing Accuracy =",accuracy_score(y_test, model.predict(normalized_X_test)))
print(encoder) 

from sklearn.metrics import f1_score
print("F1 Score =", f1_score(y_test, model.predict(normalized_X_test), average = 'macro'))

pred_list = model.predict(normalized_X_test)
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

# %%

