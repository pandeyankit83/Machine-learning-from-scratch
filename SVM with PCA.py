


import numpy as np
import pandas as pd
# from copy import deepcopy
from tqdm import tqdm

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


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


# In[5]:


def PCA_transform(k,X_train,X_valid,X_test):
#     k=5
    W = np.cov(X_train.T)
    eigval,eigvec = np.linalg.eig(W)

    print eigval
    print eigval*100/sum(eigval)
#     print np.argsort(eigval)[-k:]
    top_eigvec = eigvec[:,np.argsort(eigval)[-k:]]
#     print top_eigvec
    print np.argsort(eigval)
    X_means = np.mean(X_train.T,axis=1).reshape(60,1)
    avg_X_train = X_train.T - X_means
    avg_X_valid = X_valid.T - X_means
    avg_X_test = X_test.T - X_means
    
    return np.dot(top_eigvec.T,avg_X_train).T,np.dot(top_eigvec.T,avg_X_valid).T,np.dot(top_eigvec.T,avg_X_test).T
PCA_transform(1,X_train,X_valid,X_test)[0].shape


# In[4]:


def primal_svm(X, Y, C = 1.0):

    n = len(X[0])
    m = len(Y)

    d = n+m+1    
    P = cvxopt_matrix(0.0, (d, d))
    for i in range(n):
        P[i,i] = 1.0
    
    q = cvxopt_matrix(0.0,(d,1))
    for i in range(n,n+m):
        q[i] = C
    q[-1] = 0.0

    h = cvxopt_matrix(-1.0,(m+m,1))
    h[m:] = 0.0

#     print m
    G = cvxopt_matrix(0.0, (2*m,d))
    for i in range(m):
        G[i,:n] = -Y[i] * X[i]
        G[i,n+i] = -1
        G[i,-1] = -Y[i]

    for i in range(m,m+m):
        G[i,n+i-m] = -1.0
        
    cvxopt_solvers.options['show_progress'] = False
    vars = cvxopt_solvers.qp(P,q,G,h)['x']
    w = vars[0:n]
    b = vars[-1]

    return w,b
    

def acc(w,b,X,Y):
    Y_pred = np.dot(X,w)+b
#     print y_pred.shape
    Y_pred[Y_pred>0]=1
    Y_pred[Y_pred<0]=-1
    return sum(Y_pred == Y)*1.0/len(Y_pred)



# In[5]:


for k in range(1,7):
    for j in range(0,4):
        C = 10**j
        
        X_train_pca,X_valid_pca,X_test_pca = PCA_transform(k,X_train,X_valid,X_test)

        w,b = primal_svm(X_train_pca,Y_train, C)

        print k,C, acc(w,b,X_train_pca,Y_train), acc(w,b,X_valid_pca,Y_valid), acc(w,b,X_test_pca,Y_test)


# In[35]:


# for k in range(1,7):
for j in range(0,4):
    C = 10**j

#         X_train_pca,X_valid_pca,X_test_pca = PCA_transform(k,X_train,X_valid,X_test)

    w,b = primal_svm(X_train,Y_train, C)

    print k,C, acc(w,b,X_train,Y_train), acc(w,b,X_valid,Y_valid), acc(w,b,X_test,Y_test)






