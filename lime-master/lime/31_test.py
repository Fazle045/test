import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime_tabular
from discretize import QuartileDiscretizer
from discretize import DecileDiscretizer
from discretize import EntropyDiscretizer
from discretize import BaseDiscretizer
from discretize import StatsDiscretizer
import explanation
import lime_base
import scipy as sp
from sklearn.metrics import pairwise_distances

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#import lime_tabular

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *

#-- Pytorch specific libraries import -----#
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#### dataset Import ###########
df = pd.read_csv("dataset_phishing.csv")

# Create a LabelEncoder object
le = LabelEncoder()
# Fit and transform the status column using the LabelEncoder object
df['target'] = le.fit_transform(df['status'])       ########## Legitimate =0 and fishing = 1
df[['url','target']].head(5)
df=df.drop('status', axis=1)
print(df.shape)

#Train & Test Set
X= df.iloc[: , 1:-1]
#y = upsampled_df['Churn']
y= df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)


likely_cat = {}
for var in X.iloc[:,:].columns:
    likely_cat[var] = 1.*X[var].nunique()/X[var].count() < 0.002 

num_cols = []
cat_cols = []
for col in likely_cat.keys():
    if (likely_cat[col] == False):
        num_cols.append(col)
    else:
        cat_cols.append(col)

cat_cols_index=[]
num_cols_index=[]
j=0
for i in X:
  if i in cat_cols:
    cat_cols_index.append(j)
    j=j+1
  else:
    num_cols_index.append(j)
    j=j+1

#print(num_cols_index)

###First use a MinMaxscaler to scale all the features of Train & Test dataframes

scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled =  scaler.fit_transform(X_test.values)



###Then convert the Train and Test sets into Tensors

x_tensor =  torch.from_numpy(X_train_scaled).float()
y_tensor =  torch.from_numpy(y_train.values.ravel()).float()
xtest_tensor =  torch.from_numpy(X_test_scaled).float()
ytest_tensor =  torch.from_numpy(y_test.values.ravel()).float()



###############                                             MODEL cREATION          ############################
###############                                                                     ############################
#Define a batch size , 
bs = 64
#Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
y_tensor = y_tensor.unsqueeze(1)
train_ds = TensorDataset(x_tensor, y_tensor)
#Pytorch’s DataLoader is responsible for managing batches. 
#You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
train_dl = DataLoader(train_ds, batch_size=bs)


#For the validation/test dataset
ytest_tensor = ytest_tensor.unsqueeze(1)
test_ds = TensorDataset(xtest_tensor, ytest_tensor)
test_loader = DataLoader(test_ds, batch_size=32)




n_input_dim = X_train.shape[1]

#Layer size
n_hidden1 = 300  # Number of hidden nodes
n_hidden2 = 100
n_output =  1   # Number of output nodes = for binary classifier


class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, n_hidden1) 
        self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
        self.layer_out = nn.Linear(n_hidden2, n_output) 
        
        
        self.relu = nn.ReLU()
        self.sigmoid =  nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
        self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
        
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))
        
        return x
    

model = ChurnModel()
#print(model)


#Loss Computation
loss_func = nn.BCELoss()
#loss_func =nn.CrossEntropyLoss()
#Optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 50


###############                                             MODEL CREATION  END          ############################
###############                                                                          ############################




##################################################################Train The Model########################################
model.train()
train_loss = []
for epoch in range(epochs):
    #Within each epoch run the subsets of data = batch sizes.
    for xb, yb in train_dl:
        y_pred = model(xb)            # Forward Propagation
        loss = loss_func(y_pred, yb)  # Loss Computation
        optimizer.zero_grad()         # Clearing all previous gradients, setting to zero 
        loss.backward()               # Back Propagation
        optimizer.step()              # Updating the parameters 
    #print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
    train_loss.append(loss.item())


print('Last iteration loss value: '+str(loss.item()))

##################################################################Prediction for test data and calculate various accuracy scores########################################

import itertools

y_pred_list = []
model.eval()
#Since we don't need model to back propagate the gradients in test set we use torch.no_grad()
# reduces memory usage and speeds up computation
with torch.no_grad():
    for xb_test,yb_test  in test_loader:
        y_test_pred = model(xb_test)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.detach().numpy())

#Takes arrays and makes them list of list for each batch        
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
#flattens the lists in sequence
ytest_pred = list(itertools.chain.from_iterable(y_pred_list))


y_true_test = y_test.values.ravel()
conf_matrix = confusion_matrix(y_true_test ,ytest_pred)
'''print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Precision of the MLP :\t"+str(precision_score(y_true_test,ytest_pred)))
print("Recall of the MLP    :\t"+str(recall_score(y_true_test,ytest_pred)))
print("F1 Score of the Model :\t"+str(f1_score(y_true_test,ytest_pred)))

accuracy = sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print("Accuracy from Confusion Matrix: {:.2f}%".format(accuracy * 100))'''


#################################################   Classification for a single instance ############################################

t=X_train.iloc[40,:].values
np.set_printoptions(precision=4, suppress=True)
t_reshaped = t.reshape(1, -1)

# fit the scaler on the training data
# assuming that X_train is your training data
scaler.fit(X_train)

# transform the single instance array
t_scaled = scaler.transform(t_reshaped)

t_scaled_ten=torch.from_numpy(t_scaled).float()

# Make the prediction
with torch.no_grad():
    y_test_pred = model(t_scaled_ten)
    y_pred_tag = torch.round(y_test_pred)

# Print the predicted class
print(int(y_pred_tag))

categorical_names={i: list(X_train.iloc[:, i].unique()) for i in cat_cols_index}

explainer = lime_tabular.LimeTabularExplainer(X_train.values,mode = 'classification', feature_names=X_train.columns, class_names=['legitimate', 'fishing'], discretize_continuous=True, training_labels=y_train.values, categorical_features=cat_cols_index, categorical_names=categorical_names, sample_around_instance=True)


data_row=X_train.iloc[44,:]
#print('gggggggggggggggggggg',data_row)
num_samples=10000
if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
    # Preventative code: if sparse, convert to csr format if not in csr format already
    data_row = data_row.tocsr()
data, inverse= explainer.data_inverse(data_row=data_row,num_samples=num_samples,sampling_method='gaussian')

fd=pd.DataFrame(inverse)
#print('hhhh',(fd))
fd.to_csv('new_phishing_data.csv', index=False)