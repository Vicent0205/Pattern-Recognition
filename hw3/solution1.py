from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
boston=load_boston()
#print(boston.DESCR)

print(boston.keys())
print(type(boston))
data=pd.DataFrame(boston.data,columns=boston.feature_names)
data['MEDV']=boston.target
#print(data[:10])
def correlation_cal():
    meanData=data.mean()
    stdData=data.std()
    #print(meanData)
    m,n=data.shape
    print(m,n)
    correlation=[]
    for col in list(data.columns):
        if(col=="MEDV"):
            continue
        else:
            num=np.mean((data.loc[:,col]-meanData[col])*(data.loc[:,'MEDV']-meanData['MEDV']))
            deno=stdData[col]*stdData['MEDV']
            correlation.append(num/deno)
    print(correlation)
    print(len(correlation))
def findPrincical():
    X=data.copy()
    X=X.drop(columns=["MEDV"])
    meanX=X.mean()
    for col in list(X.columns):
        X[col]=X[col]-meanX[col]
    X=np.array(X)

    # covar
    n=13
    covar=X.T.dot(X)/(n-1)
    c=np.linalg.eig(covar)
    #print(c)
    #print(c.shape)
    ans=[]
    tempLen=len(c[0])
    for i in range(tempLen):
        ans.append((c[0][i],c[1][i]))
    sorted(ans,key=(lambda x:x[0]),reverse=True)
    #print(ans)
    for tu in ans:
        print(tu[0])
        print(tu[1])
    return ans

def corr_principal(k):
    prin=findPrincical()
    base=[prin[0][1]]
    for i in range(1,k):
        base=np.concatenate((base,[prin[i][1]]),axis=0)
    #print(base)
    #print(base.shape)
    copyData=data.copy()
    X=copyData.drop(columns=["MEDV"])
    X=np.array(X)
    new_X=X.dot(base.T)

    mean_X=np.mean(new_X,axis=0)
    std_X=np.std(new_X,axis=0)
    new_X=new_X.T

    #print(mean_X)
    #print(new_X.shape)
    Y=np.array(copyData["MEDV"])
    std_Y=np.std(Y,axis=0)
    mean_Y=np.mean(Y,axis=0)
    Y=Y.T
    #print(mean_Y)
    #print(Y.shape)

    # calculate the Pearson linear coorelation
    ans=[]
    for i in range(k):
        num=np.mean((new_X[i]-mean_X[i])*(Y-mean_Y))
        #print(num)
        deno=std_X[i]*std_Y
        ans.append(num/deno)
        #print(deno)
    print(ans)


corr_principal(3)
#correlation_cal()
#findPrincical()
