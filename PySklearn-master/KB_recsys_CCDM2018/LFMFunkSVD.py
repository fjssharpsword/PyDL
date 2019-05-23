# -*- coding: utf-8 -*-
'''
Created on 2018年3月5日

@author: Jason.F
@summary: CCDM2018 知识项推荐-LFM
'''
import time
import os
import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from math import sqrt  
'''
参考：
http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/
'''
class FunkSVD:
    
    def __init__(self,method):
        self.method=method 
    
    """
    @INPUT:
        R     : a matrix to be factorized, dimension N x M
        P     : an initial matrix of dimension N x K
        Q     : an initial matrix of dimension M x K
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        beta  : the regularization parameter
    @OUTPUT:
        the final matrices P and Q
    """
    def matrix_factorization(self,R, P, Q, K, steps, alpha=0.0002, beta=0.002):
        Q = Q.T
        for step in xrange(steps):
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            eR = np.dot(P,Q)
            e = 0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                        for k in xrange(K):
                            e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
            if e < 0.001:
                break
        return P, Q.T


if __name__ == "__main__":    
    start = time.clock() 
    
    homedir = os.getcwd()#获取当前文件的路径
    df_kb = pd.read_csv(homedir+'/kb.csv',encoding='utf8')
    #print(df_kb['num'].value_counts())
    #del_kb=df_kb.drop_duplicates(['num'])
    df_kb = df_kb.loc[(df_kb['num']<60)]#假设一天学习两次
    #print (df_kb.shape[0])
    #阅读次数做区间缩放
    #df_kb['rating']=MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(df_kb['num'])
    #num_mean = df_kb['num'].mean()
    num_max=df_kb['num'].max()
    num_min=df_kb['num'].min()
    #df_kb['rating']=df_kb.apply(lambda x: float('%.6f' % float(x['num']/num_mean)),axis=1)#行
    #df_kb['rating']=df_kb['num'].apply(lambda x: float('%.6f' % float(x/num_mean)))
    df_kb['rating']=df_kb['num'].apply(lambda x: (x-num_min+1)*1.0/(num_max-num_min+1) )
    df_kb=df_kb[['csr','ke','rating']]
    test=df_kb.sample(frac=0.1)#抽样10%比例测试
    #建立评分矩阵
    N =len(df_kb['csr'].unique())#csr个数，评分矩阵的行数
    M =len(df_kb['ke'].unique())#ke个数，评分矩阵的行数
    '''
    dict_num={}
    for csr,ke,rating in  np.array(df_kb).tolist():
        dict_num.setdefault(csr,{})
        dict_num[csr][ke] =rating
    for csr, row in dict_num.iteritems():
        for ke, num in row.iteritems():
            Array_kb[csr,ke] = num 
    '''
    R = np.zeros((N, M))#转成R矩阵，非常稀疏
    for index, row in df_kb.iterrows(): # 获取每行的值
        R[int(row['csr'])][int(row['ke'])] = row['rating']      
    #R.fillna(0,inplace=True)  #未评分填充为0      
    #SVD分解，随机梯度下降求解P、Q
    fsvd=FunkSVD(method='FunkSVD')#传递方法
    N = len(R)
    M = len(R[0])
    print ("%3s%20s%20s" % ('K','steps','RMSE'))
    for K in [3,5,8,10]:#隐因子
        for steps in [2,100,200]:#迭代次数
            P = np.random.rand(N,K)
            Q = np.random.rand(M,K)
            #SVD分解
            nP, nQ = fsvd.matrix_factorization(R, P, Q, K,steps)
            nR = np.dot(nP, nQ.T)  
            #RMSE评估   
            squaredError = []
            for index, row in test.iterrows(): # 获取每行的值
                pRating=nR[int(row['csr'])][int(row['ke'])] #获取预测值
                TRating=row['rating']
                error=TRating-pRating
                squaredError.append(error * error)
            RMSE =sqrt(sum(squaredError) / len(squaredError))#均方根误差RMSE 
            print ("%3d%20d%20.6f" % (K,steps,RMSE))
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))
