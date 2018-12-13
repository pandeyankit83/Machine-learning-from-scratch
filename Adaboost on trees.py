

import numpy as np
import pandas as pd
import pprint
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import pickle as p


# In[36]:


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


# In[38]:


#5tress are possible

def gettrees(df_X,df_Y,col_list):
    df_x = df_X
    df_y = df_Y #vector with 4 values {+1,-1}


    cols = col_list
    # cols = [0,1,2]

    #############tree 1 ^^^^^
    tree1 = Node(cols[1],np.sum(df_y))
    tree1.left = Node(cols[0],0)
    tree1.right = Node(cols[2],0)

    tree1.left.left = Node(None,df_y[0])
    tree1.left.right = Node(None,df_y[1])
      
    tree1.right.left = Node(None,df_y[2])
    tree1.right.right = Node(None,df_y[3])
    
#     ##############tree 2 //////
    tree2 = Node(cols[1],np.mean(df_y))
    tree2.left = Node(cols[0],0)
    tree2.right = Node(None,df_y[0])
    
    tree2.left.left = Node(cols[2],0)
    tree2.left.right = Node(None,df_y[1])
    
    tree2.left.left.left = Node(None,df_y[2])
    tree2.left.left.right = Node(None,df_y[3])
        
#     ##############tree 3 <<<<
    tree3 = Node(cols[1],0)
    tree3.left = Node(cols[0],0)
    tree3.right = Node(None,df_y[0])
    
    tree3.left.left = Node(None,df_y[1])
    tree3.left.right = Node(cols[2],0)
    
    tree3.left.right.left = Node(None,df_y[2])
    tree3.left.right.right = Node(None,df_y[3])    
    

#     ##############tree 4 \\\\\\
    tree4 = Node(cols[1],np.mean(df_y))
    tree4.left = Node(None,df_y[0])
    tree4.right = Node(cols[0],0)
    
    tree4.right.left = Node(None,df_y[1])
    tree4.right.right = Node(cols[2],0)
    
    tree4.right.right.left = Node(None,df_y[2])
    tree4.right.right.right = Node(None,df_y[3])
    
#     ##############tree 5 >>>>
    tree5 = Node(cols[1],np.mean(df_y))
    tree5.left = Node(None,df_y[0])
    tree5.right = Node(cols[0],0)
    
    tree5.right.left = Node(cols[2],0)
    tree5.right.right = Node(None,df_y[1])
    
    tree5.right.left.left = Node(None,df_y[2])
    tree5.right.left.right = Node(None,df_y[3])
    return [tree1,tree2,tree3,tree4,tree5]
#     return tree2#,tree2,tree3,tree4,tree5




t = gettrees(X_train,np.asarray([1,-1,1,-1]), [0,1,2])




def predict(tree, row):
    if tree.col ==None:     
        return tree.prob
    
#     print tree.col
    if row[:,tree.col]==0:
        return predict(tree.left,row)
    else:
        return predict(tree.right,row)
    
def best_tree(H,w,X,Y):    
    e = np.zeros(len(H))
    for k in range(len(H)):
        t = H[k]
        Y_pred =Y*0
        for i in range(X.shape[0]):
            p = predict(t,X[i].reshape(1,-1))
            Y_pred[i]=p
            if Y[i]!=Y_pred[i]:
                e[k] += w[i]
                
    #for preds term in w[t+1]
    t = H[np.argmin(e)]
    Y_pred =Y*0
    for i in range(X.shape[0]):
        p = predict(t,X[i].reshape(1,-1))
        Y_pred[i]=p
    return t,np.min(e),Y_pred

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


# In[41]:


print predict(t[0],np.asarray([0,1,1]).reshape(1,-1))


# In[42]:


def read(tree):
    if tree.col ==None:     
        print "prob-->",tree.prob
        return None
    
    print "curr node-->",tree.col
    print "left"
    read(tree.left)
    print "curr node-->",tree.col
    print "right"
    read(tree.right)
read(t[0])


# ## Adaboost

#generating T = 22^3 *5 hypotheses
H=[]
m = X_train.shape[1]
labels = np.asarray(list(product([-1,1],repeat=4))).reshape(16,-1)
for i in tqdm(range(m)):
    for j in range(m):
        for k in range(m):
            cols = [i,j,k]
            for l in range(len(labels)):
#             X_sub = X_train[:,cols]
                trees = gettrees(X_train,labels[l],cols)
                for t in trees:
                    H.append(t)
print len(H)     


# In[44]:


#initializing the weight matrix
T=5
w_mat = np.zeros([Y_train.shape[0],T+1])
w_mat[:,0]=1.0/Y_train.shape[0]
m=Y_train.shape[0]
# w = w.reshape(-1,1)
# e = np.zeros(len(H))

out=[]
for t in tqdm(range(T)):
    
    #get best hypothesis
    h_t, e_t, preds = best_tree(H,w_mat[:,t],X_train,Y_train)
    alpha_t = 0.5*np.log((1.0-e_t)/e_t)
    for i in range(m): 
        w_mat[i,t+1] = (w_mat[i,t])*(np.exp((-1.0*Y_train[i]*preds[i]*alpha_t))/(2*np.sqrt(e_t*(1-e_t))))
        
    out.append((h_t,e_t,alpha_t))
print out


# In[47]:


for i in range(5):
    print out[i][1],out[i][2]
filename = "trees5.pickle"
outfile = open(filename,'wb')
p.dump(out,outfile)
outfile.close()


# In[48]:


##
# def gridsearch(T,H, X_train, Y_train, X_test,Y_test):
T=11
f = file('boost.txt',"w+")
w_mat = np.zeros([Y_train.shape[0],T+1])
w_mat[:,0]=1.0/Y_train.shape[0]
m=Y_train.shape[0]


model=[]
filename = "trees10.pickle"
out1=[]
for t in tqdm(range(T)):
    f = open('boost.txt',"a+")
    #get best hypothesis
    h_t, e_t, preds = best_tree(H,w_mat[:,t],X_train,Y_train)
    alpha_t = 0.5*np.log((1.0-e_t)/e_t)
    for i in range(m): 
        w_mat[i,t+1] = (w_mat[i,t])*(np.exp((-1.0*Y_train[i]*preds[i]*alpha_t))/(2*np.sqrt(e_t*(1-e_t))))

    model.append((h_t,alpha_t))
    out = str(t)+", "+str(e_t)+", "+str(alpha_t)+", "+str(get_acc(model,X_train,Y_train))+", "+str(get_acc(model,X_test,Y_test))
    print out    
    f.write(out)
    f.write("\n")
    f.close()
    
    out1.append([h_t,e_t,alpha_t,out])
    outfile = open(filename,'wb')
    p.dump(out,outfile)
    outfile.close()


# In[51]:


o = p.load("trees10.pickle")


# In[ ]:


f = file('boost.txt',"w")
tup = gridsearch(T,H, X_train, Y_train, X_test,Y_test)
print T,tup
f.write(str(T))
f.write(",")
f.write(str(tup[0]))
f.write(",")
f.write(str(tup[1]))
f.write("\n")
f.close()







