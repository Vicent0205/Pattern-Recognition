
from col_svm import col_svm
import pandas as pd
import numpy as np
df=pd.read_csv("data_for_student/train_unsupervise/train_unsupervise_data.csv",names=[i for i in range(3000)])
print(df.shape)
data=df.to_numpy()
data=data.T
svm=col_svm()
put_list=[0]*(3000)
#self-training
threshold=0.8
while(True):
    if data.shape[0]<=10:
        break
    y=svm.predictPure(data)
    y_label=y[0]
    y_confi=y[1]
    putIndex=np.where(y_confi>threshold)
    putData=data[putIndex]
    putY=y_label[putIndex]

    col_svm.add_data(putData,putY)

    n=y.shape[0]
    remainIndex=list(set([i for i in range(n)])-set(putIndex))
    data=data[remainIndex]
    