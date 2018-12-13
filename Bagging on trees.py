
from __future__ import division
import numpy as np
import pandas as pd
import pprint
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import pickle as p
import random
from random import randint




df_train = pd.read_csv("heart_train.data",header=None)
df_test = pd.read_csv("heart_test.data",header=None)
X_train = np.asarray(df_train[df_train.columns[1:]])
Y_train = np.asarray(df_train[[0]])
X_test = np.asarray(df_test[df_test.columns[1:]])
Y_test = np.asarray(df_test[[0]])

Y_train[Y_train==0]=-1
Y_test[Y_test==0]=-1

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape


class Node: 
  
    # Utility to create a new node 
    def __init__(self , item,prob=0): 
        self.col = item 
        self.prob = prob
        self.left = None
        self.right = None




def entropy(D):
    total=0
    m = D.shape[0]
    for val in np.unique(D):
        p = D[D[:,0]==val].shape[0]*1.0/m
        total += -1*p*np.log2(p) #-1*(1-p)*np.log2(1-p)
    return total

def info_gain(D,Y):
    m = Y.shape[0]*1.0
    total=0
    for val in np.unique(D):
        total += ((D[D[:,0]==val].shape[0]*1.0/m)*entropy(Y[D[:,0]==val]))
    return entropy(Y) - total


# In[215]:


def get_best_tree(X,Y):
    X=X
    Y=Y
    ig=np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        ig[i] = info_gain(X[:,i].reshape(-1,1),Y)
#         print i,ig[i]
    ind = np.argmax(ig)
    t = Node(ind,0)
    t.left = Node(None,1) if np.sum(Y[X[:,ind]==0])>=0 else Node(None,-1)
    t.right = Node(None,1) if np.sum(Y[X[:,ind]==1])>=0 else Node(None,-1)
    return t

def predict(tree, row):
    if tree.col ==None:     
        return tree.prob
    
#     print tree.col
    if row[:,tree.col]==0:
        return predict(tree.left,row)
    else:
        return predict(tree.right,row)
    
def get_acc(model,X,Y):
    Y_preds = Y*0
    for i in range(len(X)):
        row = X[i].reshape(1,-1)
        pred=0
        for tup in model:
            p = predict(tup[0],row)
            pred += tup[1]*p
        if pred>=0:
            Y_preds[i]=1
        else:
            Y_preds[i]=-1
    return (sum(Y_preds==Y)*1.0/len(Y))


# In[268]:


##start bagging and collect hypotheses
boot = 20
model = []
for k in range(boot):
    m = X_train.shape[0]
    row_ind=[]
    for i in range(m):
        row_ind.append(randint(0,79)) 

    t = get_best_tree(X_train[row_ind,:],Y_train[row_ind,:])
    print t.col
    model.append((t,1))
print(len(H))


print get_acc(model,X_train,Y_train)
print get_acc(model,X_test,Y_test)
    

get_best_tree(X_train,Y_train)


get_best_tree(X_train,Y_train).col






