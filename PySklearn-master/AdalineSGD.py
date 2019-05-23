# -*- coding: utf-8 -*-
'''
Created on 2017年12月21日
@author: Jason.F
@summary: 自适应线性神经网络学习算法
'''
import numpy as np
import time
import matplotlib.pyplot  as plt
import pandas as pd
from numpy.random import seed

class AdalineSGD(object):
    '''
    Adaptive Linear Neuron classifier.
    
    hyper-Parameters
    eta:float=Learning rate (between 0.0 and 1.0)
    n_iter:int=Passes over the training dataset.
    
    Attributes
    w_:ld-array=weights after fitting.
    costs_:list=Number of misclassification in every epoch.
    shuffle:bool(default:True)=Shuffles training data every epoch if True to prevent cycles.
    random_state:int(default:None)=set random state for shuffling and initializing the weights.
    '''
    def __init__(self,eta=0.01,n_iter=20,shuffle=True,random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        if random_state:
            seed(random_state)
    
    def fit(self,X,y):
        '''
        Fit training data.
        Parameters
        X:{array-like},shape=[n_samples,n_features]=Training vectors,where n_samples is the number of samples and n_features is the number of features.
        y:array-like,shape=[n_samples]=Target values.
        Returns
        self:object
        '''
        self._initialize_weights(X.shape[1])
        self.cost_=[]
        
        for i in range(self.n_iter):
            if self.shuffle:
                X,y=self._shuffle(X,y)
            cost=[]
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
            
        return self
    
    def partial_fit(self,X,y):
        #Fit training data without reinitializing the weights
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi,target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
            
        return self
    
    def _shuffle(self,X,y):
        #shuffle training data
        r=np.random.permutation(len(y))
        return X[r],y[r]
    
    def _initialize_weights(self,m):
        #Initialize weights to zeros
        self.w_ =np.zeros(1+m)
        self.w_initialized=True
    
    def _update_weights(self,xi,target):
        #apply adaline learning rule to update the weights
        output=self.net_input(xi)
        error=(target-output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]  += self.eta * error
        cost= 0.5 * error ** 2
        return cost
        
    def net_input(self,X):
        #calculate net input
        return np.dot(X,self.w_[1:])+self.w_[0]
        
    def activation(self,X):
        #computer linear activation
        return self.net_input(X)
    
    def predict(self,X):
        #return class label after unit step
        return np.where(self.activation(X)>=0.0,1,-1)       

if __name__ == "__main__":   
    start = time.clock()  
    
    #训练数据
    train =pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
    X_train = train.drop([4], axis=1).values #dataframe convert to array
    y_train = train[4].values
    #特征值标准化，特征缩放方法，使数据具有标准正态分布的特性，各特征的均值为0，标准差为1.
    X_std=np.copy(X_train)
    X_std[:,0]=(X_train[:,0]-X_train[:,0].mean()) / X_train[:,0].std()
    X_std[:,1]=(X_train[:,1]-X_train[:,1].mean()) / X_train[:,1].std()
    #X_std[:,2]=(X_train[:,2]-X_train[:,2].mean()) / X_train[:,2].std()
    #X_std[:,3]=(X_train[:,3]-X_train[:,3].mean()) / X_train[:,3].std()
    y=np.where(y_train == 'Iris-setosa',-1,1)#one vs rest:OvR
    
    #学习速率和迭代次数者两个超参进行观察
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(8,4))
    #eta=0.01,n_iter=20
    agd1 = AdalineSGD(eta=0.01,n_iter=20,random_state=1).fit(X_std,y)
    print (agd1.predict([6.9,3.0,5.1,1.8]))#预测
    ax[0].plot(range(1,len(agd1.cost_)+1),agd1.cost_,marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Average Cost')
    ax[0].set_title('Adaline-learning rate 0.01')
    #eta=0.0001,n_iter=20
    agd2 = AdalineSGD(eta=0.0001,n_iter=20,random_state=1).fit(X_std,y)
    print (agd2.predict([6.9,3.0,5.1,1.8]))#预测
    ax[1].plot(range(1,len(agd2.cost_)+1),agd2.cost_,marker='x')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Average Cost')
    ax[1].set_title('Adaline-learning rate 0.0001')
    #show 
    plt.show()
    
    #测试在线更新
    print (agd2.w_) #更新前
    agd2.partial_fit(X_std[0,:],y[0])
    print (agd2.w_) #更新后
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))    
    
        
