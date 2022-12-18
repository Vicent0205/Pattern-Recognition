import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
dataGlo=pd.read_csv('fisheriris.data',sep=',',names=['sepal_lengh_cm','sepal_width_cm','petal_length_cm','petal_width_cm','class'])
data=dataGlo.copy()
data=data.drop(columns="class")
#print(data[:10])
print(data.shape[1])
m,n=data.shape
random.seed(5)
def isChange(arr1,arr2):
    #print(arr1)
    #print(arr2)
    for i in range(4):
        if(arr1[i]!=arr2[i]):
            return True
    return False
def calDis(i, centerList,j):
    #we choose the Euclidean distance
    currNode=np.array(data.loc[[i],])
    return np.linalg.norm(currNode-centerList[j])
def kMeans(k):
    #随机选k个点作为中心点
    #belongList=[0]*m
    chooseList=[i for i in range(m)]
    temp=random.sample(chooseList,k)
    centerList=[]
    curr=1
    for index in temp:
        #belongList[index]=curr
        curr+=1
        centerList.append(np.array(data.loc[[index],]).reshape(4,))
    #print(centerList)

    change=True
    while change:
        # find every node's belong
        category=[]
        for i in range(k):
            category.append([])
        #print(category)
        for i in range(m):
            minDis=calDis(i,centerList,0)
            minIndex=0
            for j in range(1,k):
                currDis=calDis(i,centerList,j)
                if(currDis<minDis):
                    minDis=currDis
                    minIndex=j
            #print("minIndex "+str(minIndex))
            category[minIndex].append(i)

        change=False
        # update the center
        for i in range(k):
            newFrame=data.loc[category[i],]
            newMean=np.array(newFrame.mean())
            if(isChange(newMean,centerList[i])):
                change=True
            centerList[i]=newMean
            #print(newMean)
    return centerList
def test_accuracy(k):
    centerList=kMeans(k)
    print(centerList)
    category=[]
    for i in range(k):
        category.append([])
    for i in range(m):
        minDis=calDis(i,centerList,0)
        minIndex=0
        for j in range(1,k):
            currDis=calDis(i,centerList,j)
            if(currDis<minDis):
                minDis=currDis
                minIndex=j
        #print("minIndex "+str(minIndex))
        category[minIndex].append(i)
    
    for i in range(k):
        print("size")
        print(len(category[i]))
        print("i "+str(i))
        classNum=[0,0,0]
        for rowIndex in category[i]:
            if(dataGlo.at[rowIndex,"class"]=="Iris-setosa"):
                classNum[0]+=1
            elif(dataGlo.at[rowIndex,"class"]=="Iris-versicolor"):
                classNum[1]+=1
            elif(dataGlo.at[rowIndex,"class"]=="Iris-virginica"):
                classNum[2]+=1
        print("num0  "+str(classNum[0]))
        print("num1  "+str(classNum[1]))
        print("num2  "+str(classNum[2]))
    color=["red","green","blue","black","yellow"]
    for i in range(k):
        sub=dataGlo.loc[category[i],]
        plt.scatter(sub.loc[:,"petal_length_cm"],sub.loc[:,"petal_width_cm"],c=color[i])
    plt.xlabel("petal_length_cm")
    plt.ylabel("petal_width_cm")
    plt.legend(loc=2)
    plt.show()
test_accuracy(5)

    
