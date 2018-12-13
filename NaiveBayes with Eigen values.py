

from __future__ import division
import numpy as np
import pandas as pd
import scipy
# from copy import deepcopy
from tqdm import tqdm

from random import randint
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.preprocessing import PolynomialFeatures



df_train = pd.read_csv("sonar_train.csv",header=None)
df_valid = pd.read_csv("sonar_valid.csv",header=None)
df_test = pd.read_csv("sonar_test.csv",header=None)
X_train = np.asarray(df_train[df_train.columns[0:-1]])
Y_train = np.asarray(df_train[df_train.columns[-1]]).reshape(-1,1)
X_valid = np.asarray(df_valid[df_valid.columns[0:-1]])
Y_valid = np.asarray(df_valid[df_train.columns[-1]]).reshape(-1,1)
X_test = np.asarray(df_test[df_test.columns[0:-1]])
Y_test = np.asarray(df_test[df_train.columns[-1]]).reshape(-1,1)

Y_train[Y_train==1]=-1
Y_valid[Y_valid==1]=-1
Y_test[Y_test==1]=-1
Y_train[Y_train==2]=1
Y_valid[Y_valid==2]=1
Y_test[Y_test==2]=1

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)
print(X_test.shape)
print(Y_test.shape)


# In[4]:


def NaiveMat(X_train,Y_train):
    m,n = X_train.shape
    labels = np.unique(Y_train)
    params = np.zeros([len(labels),m,2]) #3d array with 
    for i in range(n):
        for j,l in enumerate(labels):
            filt_col = X_train[Y_train.flatten()==l][:,i]
            params[j,i,0] = np.mean(filt_col)
            params[j,i,1] = np.sqrt(np.var(filt_col))#,ddof=1))
    return params
            

    
def NaivePredict(X_train,Y_train,X_test):
    params = NaiveMat(X_train,Y_train)
    Y_pred = np.zeros([X_test.shape[0],1])
    labels,counts=np.unique(Y_train,return_counts=True)

    priors = counts/len(Y_train)

    for i in range(X_test.shape[0]):
        row = X_test[i].reshape(1,-1)
        probs = priors.copy() #initializing with prior
        for j in range(row.shape[1]):
            for k,l in enumerate(labels):

                probs[k] = probs[k]*scipy.stats.norm.pdf(row[:,j], loc=params[k,j,0], scale=params[k,j,1])

        Y_pred[i] = labels[np.argmax(probs)]
    return Y_pred


def acc(Y, Y_):
    return sum(Y==Y_)/len(Y)
Y_pred = NaivePredict(X_train,Y_train,X_train)
print acc(Y_train, Y_pred)
Y_pred = NaivePredict(X_train,Y_train,X_test)
print acc(Y_test, Y_pred)
Y_pred = NaivePredict(X_train,Y_train,X_valid)
print acc(Y_valid, Y_pred)





def PCA_dist(k,X_train):
    W = np.cov(X_train.T)
    eigval,eigvec = np.linalg.eig(W)
    top_eigvec = eigvec[:,np.argsort(eigval)[-k:]]
    top_eigvec = top_eigvec*top_eigvec/k
    pi_dist=np.sum(top_eigvec,axis=1)

    return pi_dist#.reshape(1,-1)




m,n = X_train.shape
for k in range(1,11):
    for s in range(1,21):

        pi = PCA_dist(k,X_train)
        
        train_acc=[]
        valid_acc=[]
        test_acc=[]
        for i in range(0,100):
            cols = list(set([ np.random.choice(np.arange(0, n), p=pi) for x in range(s)]))
            Y_pred = NaivePredict(X_train[:,cols],Y_train,X_train[:,cols])
            train_acc.append(acc(Y_train, Y_pred))

            Y_pred = NaivePredict(X_train[:,cols],Y_train,X_valid[:,cols])
            valid_acc.append(acc(Y_valid, Y_pred))

            Y_pred = NaivePredict(X_train[:,cols],Y_train,X_test[:,cols])
            test_acc.append(acc(Y_test, Y_pred))

        print k,s,np.mean(train_acc),np.mean(valid_acc),np.mean(test_acc)
    
    

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, Y_train.flatten()).predict(X_test)
print gnb.score(X_test,Y_test.flatten())
print acc(Y_test,y_pred.reshape(-1,1))



def likelihood(feature, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp((-(feature - mean) ** 2) / (2 * variance))
print likelihood(0,3,4)
print scipy.stats.norm.pdf(0,loc=3,scale=2)







