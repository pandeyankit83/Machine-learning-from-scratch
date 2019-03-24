


from __future__ import division
import numpy as np
import pandas as pd
import scipy
# from copy import deepcopy
from tqdm import tqdm

from random import randint
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.special import expit




df_train = pd.read_csv("park_train.data",header=None)
df_valid = pd.read_csv("park_validation.data",header=None)
df_test = pd.read_csv("park_test.data",header=None)

X_train = np.asarray(df_train[df_train.columns[1:]])
Y_train = np.asarray(df_train[[0]])
Y_train[Y_train==0]=-1

X_valid = np.asarray(df_valid[df_valid.columns[1:]])
Y_valid = np.asarray(df_valid[[0]])
Y_valid[Y_valid==0]=-1

X_test = np.asarray(df_test[df_test.columns[1:]])
Y_test = np.asarray(df_test[[0]])
Y_test[Y_test==0]=-1

print X_train.shape
print Y_train.shape
print X_valid.shape
print Y_valid.shape
print X_test.shape
print Y_test.shape


# In[95]:


def Log_reg(X_train,Y_train,gamma=0.0001,l1=0,l2=0):
    m,n = X_train.shape
    Y_pred = np.zeros(m).reshape(-1,1)
#     w = np.random.uniform(size=[n,1])
    w = np.random.normal(loc=0,scale=10,size=[n,1])
#     w = np.zeros([n,1])+0.02
    b=1
    Y=Y_train.copy()
    obj_old = 0
    converged = False
    while not converged:

        Y_pred = np.dot(X_train,w)+b
        grad_b = np.sum((((Y+1)/2) - expit(Y_pred)),axis=0)
        grad_w = np.sum(X_train*(((Y+1)/2) -  expit(Y_pred)),axis=0).reshape(-1,1)
        b = b + gamma*(grad_b) #-l1/2 -l2*b)
        w = w + gamma*(grad_w - l1/2 - l2*w)
#         print np.exp(np.dot(X_train,w)+b)
        obj = np.sum(((Y+1)/2)*(np.dot(X_train,w)+b) - np.log(1+ np.exp(np.dot(X_train,w)+b))) - l1*(np.sum(np.absolute(w),axis=0)/2)- l2*(np.dot(w.T,w))/2

        if np.allclose(obj, obj_old,atol=1e-5):
            converged=True
        obj_old = obj
        gamma = gamma/1.01
    return w,b

def Log_pred(X,w,b):
    Y_pred = np.dot(X,w)+b
    Y_pred[Y_pred>=0]=1
    Y_pred[Y_pred<0]=-1
    return Y_pred

def acc(Y, Y_pred):
    return(sum(Y==Y_pred)*1.0/len(Y))

w,b = Log_reg(X_train,Y_train,l1=0,l2=0)
print w,b
acc(Y_test,Log_pred(X_test,w,b))





np.random.seed(42)
for i in [0.01,0.1,0,1,10,100,1000]:
    outs=[]
    for k in range(20):
        w,b = Log_reg(X_train,Y_train,l1=i,l2=0)
        outs.append([i,0,acc(Y_train,Log_pred(X_train,w,b)), acc(Y_valid,Log_pred(X_valid,w,b)),acc(Y_test,Log_pred(X_test,w,b))])
    print np.average(outs,axis=0)

for j in [0.01,0.1,0,1,10,100,1000]:
    outs=[]
    for k in range(20):
        w,b = Log_reg(X_train,Y_train,l1=0,l2=j)
        outs.append([j,0,acc(Y_train,Log_pred(X_train,w,b)), acc(Y_valid,Log_pred(X_valid,w,b)),acc(Y_test,Log_pred(X_test,w,b))])
    print np.average(outs,axis=0)






