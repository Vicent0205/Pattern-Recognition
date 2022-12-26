
from col_svm import col_svm
import pandas as pd
import numpy as np
df=pd.read_csv("data_for_student/train_unsupervise/train_unsupervise_data.csv",names=[i for i in range(3000)])
print(df.shape)
data=df.to_numpy()
data=data.T
print(data.shape)
svm=col_svm()
#put_list=[0]*(3000)
##### -----------self-training

threshold=0.8
while(True):
    if data.shape[0]<=10:
        break
    y=svm.predictPure(data)
    y_label=y[0]
    y_confi=y[1]
    print("mean "+str(np.mean(y_confi)))
    print("max "+str(np.max(y_confi)) )
    print("min "+str(np.min(y_confi)) )
    threshold=np.mean(y_confi)
    putIndex=np.where(y_confi>threshold)[0]
    putData=data[putIndex]
    putY=y_label[putIndex]
    print("put num "+str(putIndex.shape[0]))
    svm.add_data(putData,putY)

    n=y.shape[0]
    remainIndex=list(set([i for i in range(n)])-set(putIndex))
    data=data[remainIndex]
    print("data shape")
    print(data.shape)


df=pd.read_csv("data_for_student/train/train_data.csv",names=[i for i in range(1500)])
df_y=pd.read_csv("data_for_student/train/label.csv",names=[i for i in range(1500)])
#print(df.shape)
#print(df_y.shape)
data=df.to_numpy()
y=df_y.to_numpy()
data=data.T
y=y.T
y=y.reshape((y.shape[0],))

data=data[ 1: :2]
y=y[1: :2]
print(data.shape)
print(y.shape)

svm.test_accuracy(data,y)

np.savetxt("svm0w.txt",svm.svm0.weights)
np.savetxt("svm0w.txt",[svm.svm0.bias])
np.savetxt("svm1w.txt",svm.svm1.weights)
np.savetxt("svm1w.txt",[svm.svm1.bias])
np.savetxt("svm2w.txt",svm.svm2.weights)
np.savetxt("svm2w.txt",[svm.svm2.bias])
