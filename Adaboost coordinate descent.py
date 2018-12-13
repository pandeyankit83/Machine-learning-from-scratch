import numpy as np
import pandas as pd
import pprint
from copy import deepcopy
from tqdm import tqdm
from itertools import product
import pickle as p


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


# In[22]:


class Node: 
  
    # Utility to create a new node 
    def __init__(self , item,prob=0): 
        self.col = item 
        self.prob = prob
        self.left = None
        self.right = None


# In[23]:


def predict(tree, row):
    if tree.col ==None:     
        return tree.prob
    if row[:,tree.col]==0:
        return predict(tree.left,row)
    else:
        return predict(tree.right,row)

def eval_H(H, X, Y):
    Y_H = np.zeros([X.shape[0],len(H)]) 
    for k in tqdm(range(len(H))):
        t = H[k]
        Y_pred =Y*0
        for i in range(X.shape[0]):
            p = predict(t,X[i].reshape(1,-1))
#             Y_pred[i]=p
            Y_H[i,k] = p
    return Y_H

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


# In[ ]:





# In[28]:


#generating T = 22*4 hypotheses
H=[]
m = X_train.shape[1]
labels = np.asarray(list(product([-1,1],repeat=2))).reshape(4,-1)
for i in tqdm(range(m)):
    for l in range(len(labels)):
        t = Node(i,0)
        t.left = Node(None,labels[l,0])
        t.right = Node(None,labels[l,1])
        H.append(t)
    
print len(H)  


# In[29]:


Y_H = eval_H(H,X_train,Y_train)
print(Y_H.shape)


def loss_fn(Y_H,Y,alphas):
    m = len(Y)
    loss=0
    for j in range(m):
        loss += np.exp(-1.0*Y[j]*np.sum(alpha[0,:]*Y_H[j,:]))
    return loss


# In[31]:


alpha_out = np.random.random_sample(len(H),).reshape(1,-1)
loss_out=0
m = len(Y_train)
for n in range(100000):
    
    alpha = alpha_out.copy()#np.zeros(len(H)).reshape(1,-1)    
    model = []
    
    
    for t in range(len(H)):
        num = 0.
        den = 0.
        cols = range(len(H))
        cols.pop(t)
        for j in range(m):
            temp = np.exp(-1.0*Y_train[j]*np.sum(alpha[0,cols]*Y_H[j,cols]))
            if Y_H[j,t]== Y_train[j]:
                num+= temp
            else:
                den+=temp
#         print n,t,num,den
        alpha[0,t] = 0.5*np.log(num*1.0/den)
        model.append((H[t],alpha[0,t]))
    alpha_out = alpha
    
    #####calc derivative of loss
    der_loss = np.zeros(len(H))
    for t in range(len(H)):
        temp=0
        for j in range(m):
            temp += Y_train[j]*Y_H[j,t]*np.exp(-1.0*Y_train[j]*np.sum(alpha[0,:]*Y_H[j,:]))
        der_loss[t]=temp
    der_loss = np.sum(der_loss*der_loss)
        


    if n%10==0:
        acc_train = get_acc(model,X_train,Y_train)
        acc_test = get_acc(model,X_test,Y_test)
        print n,np.sum(alpha-alpha_out),der_loss,acc_train,acc_test
    if der_loss<=1e-4:
        print "converged"
        print n,np.sum(alpha-alpha_out),der_loss,acc_train,acc_test
        break

print alpha_out
print loss_fn(Y_H,Y_train,alpha_out)



print np.sum(Y_train!=Y_train)


# # Adaboost


#initializing the weight matrix
T=21
w_mat = np.zeros([Y_train.shape[0],T+1])
w_mat[:,0]=1.0/Y_train.shape[0]
m=Y_train.shape[0]

model=[]
out1=[]
for t in range(T):
    #get best hypothesis
    h_t, e_t, preds = best_tree(H,w_mat[:,t],X_train,Y_train)
    alpha_t = 0.5*np.log((1.0-e_t)/e_t)
    for i in range(m): 
        w_mat[i,t+1] = (w_mat[i,t])*(np.exp((-1.0*Y_train[i]*preds[i]*alpha_t))/(2*np.sqrt(e_t*(1-e_t))))

    model.append((h_t,alpha_t))
    out = str(t)+", "+str(e_t)+", "+str(alpha_t)+", "+str(get_acc(model,X_train,Y_train))+", "+str(get_acc(model,X_test,Y_test))
    print out    

    
    out1.append([h_t,e_t,alpha_t,out])










