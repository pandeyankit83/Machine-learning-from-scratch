import numpy as np
import pandas as pd
import pprint
from copy import deepcopy

df_train = pd.read_csv("mush_train.data",header=None)
df_test = pd.read_csv("mush_test.data",header=None)
X_train = np.asarray(df_train[df_train.columns[1:]])
Y_train = np.asarray(df_train[[0]])
X_test = np.asarray(df_test[df_test.columns[1:]])
Y_test = np.asarray(df_test[[0]])
print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

def entropy(D):
    total=0
    m = D.shape[0]
#     p = D[D[:,0]==D[0,0]].shape[0]*1.0/D.shape[0]
    for val in np.unique(D):
        p = D[D[:,0]==val].shape[0]*1.0/m
        total += -1*p*np.log2(p) #-1*(1-p)*np.log2(1-p)
    return total
# print entropy(Y_train)

def info_gain(D,Y):
    m = Y.shape[0]*1.0
    total=0
    for val in np.unique(D):
#         print val
#         print entropy(Y[D[:,0]==val])
        total += ((D[D[:,0]==val].shape[0]*1.0/m)*entropy(Y[D[:,0]==val]))
    return entropy(Y) - total

class Tree:
    value = 0
    x_cols = []
    y_counts = 0
    children = {}
    def __init__(self, value,y_counts=None, x_cols = [],children={}):
        self.value = value #index/name of col in X
        self.y_counts = y_counts
        self.x_cols = x_cols
        self.children = children


def build_dtree(X,Y):
    X=X
    Y=Y
    children={}
    gain,ind = split(X,Y)
    
#     print gain,ind
    
    if gain == 0:
#         print "leaf---", Y.shape
        return Tree(Y[0,0],None,None,None)
   
    vals = np.unique(X[:,ind])
    
    unique, counts =np.unique(Y, return_counts=True) 
    y_counts = dict(zip(unique,counts))
    
#     print "vals--->",vals
#     dtree = Tree(ind,y_counts)
    for val in vals:

        X_temp = X
        Y_temp = Y
        
        filt = X_temp[:,ind]==val

#         print X_temp[filt].shape
#         print "filter on -- ",val
#         print "X---",X_temp[filt].shape
#         print "Y---",Y_temp[filt].shape
#         print "going into--->",val
#         print "keys at current node-->",children.keys(),len(children.keys()) 
#         print "vals at current node--->",vals
        children[val] = build_dtree(X_temp[filt],Y_temp[filt])
#         print "keys at current node-->",children.keys(),len(children.keys()) 
#         print "aslkdjfdsaklf"    
    return Tree(ind,y_counts,X_cols,children)

def read_tree(tree):
    
    if tree.children==None:
        print "decision=",tree.value
        return None
    print "  ",tree.value
    print tree.children.keys()
    for child in tree.children.keys():
        print child
        read_tree(tree.children[child])
    return None
read_tree(dt)


def predict(tree,row):
#     print tree.children
    if tree.children==None:
        return tree.value
    found=False
    
    curr_ind = tree.value
    for val in tree.children.keys():
#         print val
        if val==row[0,curr_ind]:
#             print val
            found=True
            return predict(tree.children[val],row)
    if found==False:
        return max(tree.y_counts,key = tree.y_counts.get)


Y_pred = Y_train*0
for i in range(0,len(X_train)):
    Y_pred[i,0]=predict(dt,X_train[i,:].reshape(1,-1))
    
print sum(Y_pred == Y_train)/len(Y_pred == Y_train)

Y_pred = Y_test*0
for i in range(0,len(X_test)):
    Y_pred[i,0]=predict(dt,X_test[i,:].reshape(1,-1))
print sum(Y_pred == Y_test)/len(Y_pred == Y_test)


#########################################################################################################################
from sklearn.cross_validation import train_test_split

df_train = pd.read_csv("mush_train.data",header=None)
df_test = pd.read_csv("mush_test.data",header=None)
X_train = np.asarray(df_train[df_train.columns[1:]])
Y_train = np.asarray(df_train[[0]])
X_test = np.asarray(df_test[df_test.columns[1:]])
Y_test = np.asarray(df_test[[0]])
print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

X_master = np.concatenate([X_train,X_test])
Y_master = np.concatenate([Y_train,Y_test])
print X_master.shape
print Y_master.shape

f = file('temp.txt','w')
for j in range(1,20):
    s = j*0.05
    X_train, X_test,Y_train,Y_test = train_test_split(X_master,Y_master, test_size = s)

    t = build_dtree(X_train,Y_train)
    temp=[str(s)]
    Y_pred = Y_train*0    
    for i in range(0,len(X_train)):
        Y_pred[i,0]=predict(t,X_train[i,:].reshape(1,-1))

    temp.append(str(sum(Y_pred == Y_train)*1.0/len(Y_pred == Y_train)))

    Y_pred = Y_test*0
    for i in range(0,len(X_test)):
        Y_pred[i,0]=predict(t,X_test[i,:].reshape(1,-1))
    temp.append(str(sum(Y_pred == Y_test)*1.0/len(Y_pred == Y_test)))
    f.write(" ".join(temp))
    f.write("\n")
    
f.close()
