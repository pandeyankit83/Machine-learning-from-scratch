import numpy as np
import pandas as pd
import math
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.preprocessing import PolynomialFeatures

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

################################################################################33
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

    vars = cvxopt_solvers.qp(P,q,G,h)['x']
    w = vars[0:n]
    b = vars[-1]
#     print w
#     print b
    return w,b
    
print len(svm(X_train,Y_train,1))

###########################################################################################################################
def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def SVM(X,Y,X_,C,sigma):
    m, n = X.shape
    X= X*1.
    y = Y*1.
    
    # Gram matrix
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i,j] = gaussian_kernel(X[i], X[j],sigma)
    
    P = cvxopt_matrix(np.outer(y,y) * K)
#     print P.shape
#     print np.outer(y,y) * K
    q = cvxopt_matrix(np.ones(m) * -1)
#     A = cvxopt.matrix(y, (1,m))
#     b = cvxopt.matrix(0.0)
    

    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    #Setting solver parameters (change default to decrease tolerance) 
    cvxopt_solvers.options['show_progress'] = False

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

#     w = np.dot((y * alphas).T, X).reshape(-1,1)

    #alphas are non zeros for the support vectors
    S = ((alphas > 1e-2) & (alphas < C)).flatten()
   
#     #Computing b
#     K = np.zeros((m, 1))
#     for i in range(m):
#         K[i,0] = gaussian_kernel(X[i], X[0],sigma)
#     b = y - np.dot((alphas*y).T,K)
    
    #Computing b
    m_ = X[S].shape[0]
#     print len(S)
#     print S
#     print X[S].shape
    K = np.zeros((m, m_))
    for i in range(m):
        for j in range(m_):
            K[i,j] = gaussian_kernel(X[i], X[S][j],sigma)
#     print K.shape
#     print np.dot((alphas*y).T,K).shape
    b = y[S] - np.dot((alphas*y).T,K).T
    b = (max(b[y[S]==-1])+min(b[y[S]==1]))/2
    
    #computing y_pred
    m_ = X_.shape[0]
    K = np.zeros((m, m_))
    for i in range(m):
        for j in range(m_):
            K[i,j] = gaussian_kernel(X[i], X_[j],sigma)
            
    Y_pred = np.dot((alphas*y).T,K).T + b#[0]
    
    Y_pred[Y_pred>=0]=1
    Y_pred[Y_pred<0]=-1
    print b
    return Y_pred

###################################################################################################################################
def acc(w,b,X,Y):
    Y_pred = np.dot(X,w)+b
#     print y_pred.shape
    Y_pred[Y_pred>0]=1
    Y_pred[Y_pred<0]=-1
    return sum(Y_pred == Y)*1.0/len(Y_pred)

for i in range(0,9):
    C = 10**i
    w,b = primal_svm(X_train,Y_train, C)
#     w=w.T
#     print w.shape
    print C, acc(w,b,X_train,Y_train), acc(w,b,X_valid,Y_valid), acc(w,b,X_test,Y_test)

###################################################################################################################################

for i in range(0,9):
    C = 10**i
    for j in range(-1,4):
        sigma = 10**j
        temp = [C,sigma]
        y_pred = SVM(X_train,Y_train,X_train, C,sigma)
        acc=sum(y_pred==Y_train)*1.0/y_pred.shape[0]
        temp.append(float(acc))
        
        y_pred = SVM(X_train,Y_train,X_valid, C,sigma)
        acc =sum(y_pred==Y_valid)*1.0/y_pred.shape[0]
        temp.append(float(acc))
        
        y_pred = SVM(X_train,Y_train,X_test, C,sigma)
        acc=sum(y_pred==Y_test)*1.0/y_pred.shape[0]
        temp.append(float(acc))
        print temp