
from __future__ import division
import numpy as np
import pandas as pd
import scipy
# from copy import deepcopy
from tqdm import tqdm
# from itertools import product
# import pickle as p
import random
from random import randint
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal



df = pd.read_csv("leaf.data",header=None)
X = np.asarray(df[df.columns[1:]])
Y = np.asarray(df[df.columns[0]]).reshape(-1,1)

##preprocess
X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
print X
print(X.shape)
print(Y.shape)





def kmpp(k,X):
    m,n = X.shape
    c = np.zeros([k,n])
    c[0,:] = X[random.randint(0,m-1),:]
    D = np.zeros([m,k])
    for j in range(k-1):
        z= X - c[j,:].reshape(1,-1)
        D[:,j] = np.sum(z**2,axis=1)
        D[:,j] = D[:,j]/np.sum(D[:,j],axis = 0)
		probs = np.amin(D,axis =1)/sum(np.amin(D,axis =1))
        c[j+1,:] = X[np.random.choice(range(0,m),p = probs.flatten()),:]
    return c



def kmeans(k,X,init = "rand"):
    m,n = X.shape
#     c = 6*np.random.uniform(size=[k,n])-3
    if init=="rand":
        c = 6*np.random.uniform(-3,3,size=[k,n])
    else:
        c = kmpp(k,X)
    Y = np.zeros([m,1])
    Y_prev=Y+1
    max_iter=2000
    while(np.array_equal(Y_prev,Y)==False and max_iter>0):
        max_iter -=1
        Y_prev=Y.copy()
        D = np.zeros([m,k])
        for i in range(c.shape[0]):
            z= X - c[i,:].reshape(1,-1)
#             print z.shape
#             print np.sum(z**2,axis=1).shape
            D[:,i]=np.sum(z**2,axis=1)#.reshape(-1,1)
#         print D
        Y = np.argmin(D,axis=1).reshape(-1,1)
#         print c
#         labels=np.unique(Y)
#         for label in labels:
#             c[label,:] = np.average(X[Y.flatten()==label,:],axis=0)
        for i in range(k):
#             c[i,:] = np.average(X[Y.flatten()==i,:],axis=0)
#             if len(np.where(Y.flatten()==i)[0])==0:
            if i not in Y.flatten():
                c[i,:]=0
            else:
#                 print np.average(X[np.where(Y.flatten()==i),:],axis=1).shape
                c[i,:] = np.average(X[np.where(Y.flatten()==i)[0],:],axis=0)
#                 print c[i,:].shape
    return Y

def kmeans_obj(k,X,Y):
    m,n = X.shape
    labels=np.unique(Y)
#     c = np.zeros([len(labels),n])
    out=0
#     for label in labels:
    for i in range(k):
        if i in labels: 
#         center = np.average(X[Y.flatten()==label,:],axis=0)
            center = np.average(X[np.where(Y.flatten()==i)[0],:],axis=0)
            z= X[Y.flatten()==i,:] - center.reshape(1,-1)
            out+= np.sum(z**2)
    return out

Y_= kmeans(12,X)
print kmeans_obj(12,X,Y_)
Y1 = KMeans(n_clusters=12,init='random').fit_predict(X)
print kmeans_obj(12,X,Y1)




ks = [12,18,24,36,42]
for k in ks:
    outs=[]
    for i in range(20):
        outs.append(kmeans_obj(k,X,kmeans(k,X)))
#         outs.append(kmeans_obj(X,KMeans(n_clusters=12,init='random').fit_predict(X)))
#     print outs
    print k, np.mean(outs),np.var(outs)
    
    




###Gaussian clustering
def GMM(k,X,init="rand"):
    m,n = X.shape
    if init=="rand":
        mu = np.random.uniform(-3,3,size=[k,n])
    else:
        mu = kmpp(k,X)
    cov = np.asarray([np.identity(n)]*k)
    q = np.zeros([m,k])+1

#     lambdas = np.zeros(k)+1
    lambdas = np.asarray([1/k]*k)
    ll_old = -np.inf
    converged=False

    while converged==False:
            ##estep
        for i in range(m):
            den=0
            for j in range(k):
                pd = multivariate_normal(mu[j,:],cov[j,:,:],allow_singular=True)
                num = lambdas[j] * pd.pdf(X[i,:])
                den += num
                q[i,j] = num
            q[i,:] = q[i,:]/den

            ##mstep
        for j in range(k):
            mu[j,:] = np.sum(X*q[:,j].reshape(-1,1),axis=0)/np.sum(q[:,j])
        #     print np.diag(np.dot((X-mu[j,:]),(X-mu[j,:]).T))
        #     print (np.dot((X-mu[j,:]),(X-mu[j,:]).T)*q[:,j].reshape(-1,1)).shape
        #     cov[j,:,:] = np.sum(q[:,j].reshape(-1,1)*np.dot((X-mu[j,:]),(X-mu[j,:]).T),axis=0)/np.sum(q[:,j])
            w_cov = np.zeros([n,n])
            for i in range(m):
                w_cov +=q[i,j]*np.dot((X[i,:]-mu[j,:]).reshape(-1,1),(X[i,:]-mu[j,:]).reshape(1,-1))
            cov[j,:,:] = w_cov/np.sum(q[:,j])

            lambdas[j] = np.sum(q[:,j])/m
        cov = cov + np.asarray([np.identity(n)]*k)*0.00001
        ###likelihood function
        ll_new=0
        for i in range(m):
            num=0
            for j in range(k):
                pd = multivariate_normal(mu[j,:],cov[j,:,:],allow_singular=True)
                num += lambdas[j] * pd.pdf(X[i,:])
            ll_new+= np.log(num)

            
        if abs(ll_new-ll_old) < 1e-2:
            converged=True
            return q, ll_new
        else:
            ll_old = ll_new
#             print ll_new
    








ks = [12,18,24,36,42]
res=[]
for k in tqdm(ks):
    outs=[]
    for i in tqdm(range(20)):
        _,loss = GMM(k,X)
        outs.append(loss)
    res.append([k, np.mean(outs),np.var(outs)])
    print k, np.mean(outs),np.var(outs)
print res





# ## kmeans++




ks = [12,18,24,36,42]
for k in ks:
    outs=[]
    for i in range(20):
        outs.append(kmeans_obj(k,X,kmeans(k,X,init="kpp")))
#         outs.append(kmeans_obj(X,KMeans(n_clusters=12,init='random').fit_predict(X)))
#     print outs
    print k, np.mean(outs),np.var(outs)


# In[59]:


ks = [12,18,24,36,42]
res=[]
for k in tqdm(ks):
    outs=[]
    for i in tqdm(range(20)):
        _,loss = GMM(k,X,init="kpp")
        outs.append(loss)
    res.append([k, np.mean(outs),np.var(outs)])
    print k, np.mean(outs),np.var(outs)
print res






