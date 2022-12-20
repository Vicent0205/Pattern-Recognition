from solution1 import support_vector_machine
import numpy as np
import pandas as pd

class col_svm:  
    def __init__(self,C=10,features=600,sigma_sq=0.1,kernel="gaussian"):

        data,y=self.get_data()
        self.C=C
        self.features=features
        self.sigma_sq=sigma_sq
        self.kernel=kernel
        
        self.label=y
        self.data=data

        x0=data.copy()
        #y0=y.copy()
        y0=np.where(y==0,1,-1)
        print(y0)
        #y0=y0.reshape((y0.shape[0],))
        self.svm0=support_vector_machine(C,features,sigma_sq,kernel)
        gaussion_x=self.svm0.fit(x_train=x0,y_train=y0,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=True)
        #print("Training Accuracy = {}".format(self.svm0.evaluate(x0,y0)))

        #x1=data.copy()
        #y1=y.copy()
        y1=np.where(y==1,1,-1)
        #y1=y1.reshape((y1.shape[0],))
        #y1=y1.T
        self.svm1=support_vector_machine(C,features,sigma_sq,kernel)
        self.svm1.fit(x_train=x0, x_gaussion=gaussion_x,y_train=y1,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=False)
        #print("Training Accuracy = {}".format(self.svm1.evaluate(x1,y1)))

        #x2=data.copy()
        #y2=y.copy()
        y2=np.where(y==2,1,-1)
        #y2=y2.reshape((y2.shape[0],))
        #y2=y2.T
        self.svm2=support_vector_machine(C,features,sigma_sq,kernel)
        self.svm2.fit(x_train=x0,x_gaussion= gaussion_x,y_train= y2,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=False)
        #print("Training Accuracy = {}".format(self.svm2.evaluate(x2,y2)))

        self.get_accuracy()

    def get_data(self):
        df=pd.read_csv("data_for_student/train/train_data.csv",names=[i for i in range(1500)])
        df_y=pd.read_csv("data_for_student/train/label.csv",names=[i for i in range(1500)])
        print(df.shape)
        print(df_y.shape)
        data=df.to_numpy()
        y=df_y.to_numpy()
        data=data.T
        y=y.T
        y=y.reshape((y.shape[0],))
        return data,y
    
    def predict(self,x):
        y0=self.svm0.predictPure(x)
        y1=self.svm1.predictPure(x)
        y2=self.svm2.predictPure(x)
        ans=[]
        n=y0.shape[0]
        for i in range(n):
            temp=0
            largeNum=y0[i]
            if(y0[i]<y1[i]):
                temp=1
                largeNum=y1[i]
            if(y2[i]>largeNum):
                temp=2
            ans.append(temp)
        
        return np.array(ans)

    def predictPure(self,x):
        y0=self.svm0.predictPure(x)
        y1=self.svm1.predictPure(x)
        y2=self.svm2.predictPure(x)
        ans=[]
        ans1=[]
        n=y0.shape[0]
        for i in range(n):
            temp=0
            largeNum=y0[i]
            if(y0[i]<y1[i]):
                temp=1
                largeNum=y1[i]
            if(y2[i]>largeNum):
                temp=2
                largeNum=y2[i]
            
            ans.append(temp)
            ans1.append(largeNum)
        res=[ans,ans1]
        return np.array(res)
    
    def train_newdata(self,data,y):

        x0=data.copy()
        #y0=y.copy()
        y0=np.where(y==0,1,-1)
        print(y0)
        #y0=y0.reshape((y0.shape[0],))
        self.svm0=support_vector_machine(self.C,self.features,self.sigma_sq,self.kernel)
        gaussion_x=self.svm0.fit(x_train= x0,y_train= y0,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=True)
        #print("Training Accuracy = {}".format(self.svm0.evaluate(x0,y0)))

        #x1=data.copy()
        #y1=y.copy()
        y1=np.where(y==1,1,-1)
        #y1=y1.reshape((y1.shape[0],))
        #y1=y1.T
        self.svm1=support_vector_machine(self.C,self.features,self.sigma_sq,self.kernel)
        self.svm1.fit(x_train=x0, x_gaussion= gaussion_x,y_train= y1,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=False)
        #print("Training Accuracy = {}".format(self.svm1.evaluate(x1,y1)))

        #x2=data.copy()
        #y2=y.copy()
        y2=np.where(y==2,1,-1)
        #y2=y2.reshape((y2.shape[0],))
        #y2=y2.T
        self.svm2=support_vector_machine(self.C,self.features,self.sigma_sq,self.kernel)
        self.svm2.fit( x_train=x0, x_gaussion= gaussion_x,y_train= y2,epochs=20,print_every_nth_epoch=2,learning_rate=0.01,need_process=False)
        #print("Training Accuracy = {}".format(self.svm2.evaluate(x2,y2)))

    def add_data(self,x,y):
        self.data=np.concatenate((self.data,x),axis=0)
        self.label=np.concatenate((self.label,y),axis=0)

        self.train_newdata(self.data,self.label)

        print("add new data done")

    def get_accuracy(self):
        output=self.predict(self.data)
        ans=np.where(output==self.label,1,0)
        n=ans.shape[0]
        print("accuarcy  "+str(np.sum(ans)/n))
if __name__=='__main__':
    svm=col_svm()
    svm.get_accuracy()