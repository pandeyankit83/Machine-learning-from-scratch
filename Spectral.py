
from __future__ import division
import numpy as np
import pandas as pd
# from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




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





def gaussian(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2.0 * (sigma ** 2)))





def spectral_cluster(X,k,sigma=5.0):
    #calc A
    m,n = X.shape
    A = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            A[i,j] = gaussian(X[i,:],X[j,:],sigma)

    #calc D
    D = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i==j:
                D[i,j] = np.sum(A[i,:])

    L = D-A
    ####eigen things###
#     k=2
    eigval,eigvec = np.linalg.eig(L)
    # print np.argsort(eigval)
    top_eigvec = eigvec[:,np.argsort(eigval)[0:k]] #taking smalllest eigval
    V = top_eigvec

    ##clustering

    kmeans = KMeans(n_clusters=k, random_state=42,n_jobs=1).fit(V)
    return kmeans.labels_
# print spectral_cluster(X_train,2)





df_circs = pd.read_csv("circs.csv",header=None)
circs = np.asarray(df_circs)

for i,sigma in enumerate([0.001,0.01,0.1,1,5,10]):
    plt.subplot(2,3,i+1)
    plt.scatter(circs[:, 0], circs[:, 1], c=spectral_cluster(circs,2,sigma))
plt.show()

plt.subplot(1,2,1)
plt.scatter(circs[:, 0], circs[:, 1])
plt.subplot(1,2,2)
plt.scatter(circs[:, 0], circs[:, 1], c=KMeans(n_clusters=2, random_state=0).fit_predict(circs))#.labels_)





##for partitioning images
from matplotlib.image import imread
img = imread("bw.jpg")/255

im=img.flatten().reshape(-1,1)
# plt.imshow(im.reshape(75,100), cmap = 'gray')#, interpolation = 'bicubic')
print im.shape



def spectral_cluster2(X,k,sigma=5.0):
    #calc A
    m,n = X.shape
    A = np.zeros([m,m])
    A = np.exp(-(X-X.T)**2 / (2 * (sigma ** 2)))

            
    print A.shape

    D = np.diag(sum(A))
    L = D-A

    ####eigen things###
#     k=2
    eigval,eigvec = np.linalg.eig(L)
   
    top_eigvec = eigvec[:,np.argsort(eigval)[0:k]] #taking smalllest eigval
    V = top_eigvec

    ##clustering

    kmeans = KMeans(n_clusters=k, random_state=0).fit(V)
    return kmeans.labels_
clust= spectral_cluster2(im,2,1000000)#11...

plt.imshow(clust.reshape(75,100), cmap = 'gray')#, interpolation = 'bicubic')





plt.subplot(1,2,1)
plt.imshow(img, cmap = 'gray')#, interpolation = 'bicubic')
plt.subplot(1,2,2)
plt.imshow(KMeans(n_clusters=2, random_state=0,n_jobs=1).fit_predict(im).reshape(75,100), cmap = 'gray')#, interpolation = 'bicubic')
plt.show()


plt.imshow(KMeans(n_clusters=2, random_state=0,n_jobs=-1).fit_predict(im).reshape(75,100), cmap = 'gray')








