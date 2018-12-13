import numpy as np
import pandas as pd
import math

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


def kNN_predict(train,train_labels, test,k):
    
    preds = np.zeros(test.shape[0])

    for i in range(test.shape[0]):
        labels = np.hstack((train_labels,train_labels*0.))

        for j in range(train.shape[0]):
            labels[j,1] = np.linalg.norm(test[i].reshape(1,-1)-train[j].reshape(1,-1))
#         print labels
        labels = labels[labels[:,1].argsort()]

        if sum(labels[0:k,0])>0:
            preds[i] = 1
        else:
            preds[i] = -1
#         print preds[i]
    return preds.reshape(-1,1)
# kNN_predict(X_train,Y_train, X_test,2).shape


for k in [1,5, 11, 15, 21]:
# for k in range(1,20):
    temp = [k]
    preds  = kNN_predict(X_train,Y_train, X_train,k)
    temp.append(sum(preds==Y_train)*1.0/X_train.shape[0])
    
    preds  = kNN_predict(X_train,Y_train, X_valid,k)
    temp.append(sum(preds==Y_valid)*1.0/X_valid.shape[0])
    
    preds  = kNN_predict(X_train,Y_train, X_test,k)
    temp.append(sum(preds==Y_test)*1.0/X_test.shape[0])
    print temp

    