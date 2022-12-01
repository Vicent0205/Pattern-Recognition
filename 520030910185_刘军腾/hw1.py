"""
This file is to ii) question.
click run in vscode or pycharm is ok
The result is for m=200 and n=100
"""
from math import sqrt
import numpy as np
import random
import matplotlib.pyplot as plt  

random.seed(0)
iteration_index=0
epoch_index=0
iteration_num=[]
epoch_num=[]
train_loss=[]
valid_loss=[]
fo = open("foo.txt", "w")
def generaterW(n):
    W=np.zeros((n,1),dtype=float)
    for i in range(n):
        W[i][0]=random.gauss(0,1)
    return W
def generateX(m,n):
    X=np.zeros((m,n),dtype=float)
    for i in range(m):
        for j in range(n):
            X[i,j]=random.gauss(0,1)
    return X
def gerneateEplison(sig,m):
    eplison=np.zeros((m,1),dtype=float)
    for i in range(m):
        eplison[i,0]=random.gauss(0,sig)
    return eplison

"""
k is k fold
i is current fold index started from 0
X is data
Y is label 
"""
def get_k_fold_data(k,i,X,Y):
    fold_size=X.shape[0]//k
    x_train,y_train=None, None
    for j in range(k):
        start_index=j*fold_size
        end_index=(j+1)*fold_size
        x_part=X[start_index:end_index,:]
        y_part=Y[start_index:end_index]
        if(j==i):
            #put it into train
            x_valid=x_part
            y_valid=y_part
        elif x_train is None:
            x_train=x_part
            y_train=y_part
        else:
            x_train=np.concatenate((x_train,x_part),axis=0)
            y_train=np.concatenate((y_train,y_part),axis=0)
    return x_train,y_train,x_valid,y_valid
def get_loss_grad(x_train,y_train,tempW,gama):
    regular=gama/2*np.linalg.norm(tempW,ord=2)**2
    diff_loss=np.linalg.norm(x_train.dot(tempW)-y_train,ord=2)**2
    grad_w=gama*tempW+2*(np.transpose(x_train)).dot(x_train.dot(tempW)-y_train)
    return regular,diff_loss, grad_w

def line_search(curr_w,delt_w,x_train,y_train,gamma):
    a=0
    b=0.05
    t1_cal=False
    t2_cal=False
    delt_w=-delt_w
    while(b-a>0.001):
        if(not t1_cal):
            t1=a+(b-a)*(1-0.618)
            regular,diff_loss, _ =get_loss_grad(x_train,y_train,curr_w+t1*delt_w,gamma)
            loss_t1=regular+diff_loss
            #print("loss_t1 "+str(loss_t1))
        if(not t2_cal):
            t2=a+(b-a)*0.618
            regular,diff_loss, _ =get_loss_grad(x_train,y_train,curr_w+t2*delt_w,gamma)
            loss_t2=regular+diff_loss
            #print("loss_t2 "+str(loss_t2))
        t1_cal=False
        t2_cal=False        
        if(loss_t1<loss_t2):
            b=t2
            t2=t1
            loss_t2=loss_t1
            t2_cal=True
        else:
            a=t1
            t1=t2
            loss_t1=loss_t2
            t1_cal=True
    return b
    
def train(x_train,y_train,x_valid,y_valid,tempW,gama):
    regular,diff_loss,grad_w=get_loss_grad(x_train,y_train,tempW,gama)
    print("beforeregular "+str(regular)+"diff_loss "+str(diff_loss/x_train.shape[0]))
    global iteration_index
    global iteration_num
    global train_loss
    for t in range(100):
        regular,diff_loss,grad_w=get_loss_grad(x_train,y_train,tempW,gama)

        iteration_num.append(iteration_index)
        iteration_index+=1
        train_loss.append(regular+diff_loss)

        step_size=line_search(tempW,grad_w,x_train,y_train,gama)
        #step_size=0.001
        if(t%10==0):
            print("in process regular "+str(regular)+"  diff_loss "+str(diff_loss/x_train.shape[0]))
        #print("step_size "+str(step_size))
        tempW-=step_size*grad_w
        fo.write("tempW \n")
        fo.write(str(tempW))
        fo.write("\n")
        """print("tempW")
        print(tempW)"""
    regular,diff_loss, _ =get_loss_grad(x_valid,y_valid,tempW,gama)
    print("valid_loss regular "+str(regular)+"diff_loss "+str(diff_loss/x_valid.shape[0]))

def entry(X,Y,gama,k,num_epochs):
    size_w=X.shape[1]
    tempW=np.zeros((size_w,1),dtype=float)
    for i in range(size_w):
        tempW[i,0]=random.gauss(0,1)
    #print("tempW")
    #print(tempW)
    fo.write("tempW \n")
    fo.write(str(tempW))
    fo.write("\n")
    for epoch in range(num_epochs):
        print("epoch  "+str(epoch))
        for i in range(k):
            x_train,y_train,x_valid,y_valid=get_k_fold_data(k,i,X,Y)
            print("w shape "+str(tempW.shape))
            train(x_train,y_train,x_valid,y_valid,tempW,gama)
    
    global iteration_num
    global train_loss
    plt.plot(iteration_num,train_loss)
    plt.show()
    return tempW
def BN(X):
    (m,n)=X.shape
    X_mean=X.mean(axis=0)
    X_std=X.std(axis=0)
    newX=(X-X_mean)/X_std
    return newX
if __name__ =='__main__':
    m=200
    n=100
    X=generateX(m,n)
    print(X)
    
    print(type(X))
    W=generaterW(n)
    fo.write(str(W))
    fo.write('\n')
    eplison=gerneateEplison(sqrt(0.1),m)
    Y=X.dot(W)+eplison

    newX=BN(X)
    gama=0.01
    k=10
    num_epochs=5
    tempW=entry(newX,Y,gama,k,num_epochs)
    print((abs(tempW-W)).sum()/abs(W).sum())
    
    
