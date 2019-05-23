# -*- coding: utf-8 -*-
'''
Created on 2018年3月5日

@author: Jason.F
@summary: CCDM2018 知识项推荐-SkillLFM
'''
import time
import os
import pandas as pd
import numpy as np 
from math import sqrt

class SVDPlus:
    
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
    def matrix_factorization(self,R, P, Q, SU, SI, K, steps, alpha=0.0002, beta=0.002):
        Q = Q.T
        for step in xrange(steps):
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        eij = R[i][j] - np.dot(P[i,:]+SU[i]+SI[j,0],Q[:,j])
                        for k in xrange(K):
                            P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                            Q[k][j] = Q[k][j] + alpha * (2 * eij * (P[i][k]+SU[i]+SI[j,0]) - beta * Q[k][j])
                            SI[j,0]   = SI[j,0]   + alpha * (2 * eij * Q[k][j]   - beta * SI[j,0])#相乘发生溢出，还未解决
            eR = np.dot(P,Q)
            e = 0
            for i in xrange(len(R)):
                for j in xrange(len(R[i])):
                    if R[i][j] > 0:
                        e = e + pow(R[i][j] - np.dot(P[i,:]+SU[i]+SI[j,0],Q[:,j]), 2)
                        for k in xrange(K):
                            e = e + (beta/2) * ( pow(P[i][k]+SU[i]+SI[j,0],2) + pow(Q[k][j],2) ) 
            if e < 0.001:
                break
        return P, Q.T, SI


if __name__ == "__main__":    
    start = time.clock() 
    
    homedir = os.getcwd()#获取当前文件的路径
    df_kb = pd.read_csv(homedir+'/kb.csv',encoding='utf8')
    #阅读次数做区间缩放(0,1]
    df_kb = df_kb.loc[(df_kb['num']<60)]#假设一天学习两次
    num_max=df_kb['num'].max()
    num_min=df_kb['num'].min()
    df_kb['rating']=df_kb['num'].apply(lambda x: (x-num_min+1)*1.0/(num_max-num_min+1) )
    #建立评分矩阵
    df_kb=df_kb[['csr','ke','rating']]
    test=df_kb.sample(frac=0.1)#抽样10%比例测试
    N =len(df_kb['csr'].unique())#csr个数，评分矩阵的行数
    M =len(df_kb['ke'].unique())#ke个数，评分矩阵的行数
    R = np.zeros((N, M))#转成R矩阵，非常稀疏
    for index, row in df_kb.iterrows(): # 获取每行的值
        R[int(row['csr'])][int(row['ke'])] = row['rating']    
    #计算CSR当前技能水平
    sr_count=df_kb.groupby(['csr']).size()#学习过的知识点数量
    sr_count=sr_count.apply(lambda x: (x-sr_count.min()+1)*1.0/(sr_count.max()-sr_count.min()+1))
    sr_mean=df_kb.groupby(['csr'])['rating'].mean()
    SU = sr_count*1.0/sr_mean#每个用户的技能水平
    #SVD分解，随机梯度下降求解P、Q 
    svdp=SVDPlus(method='SVDPlus')#传递方法
    N = len(R)
    M = len(R[0])
    print ("%3s%20s%20s" % ('K','steps','RMSE'))
    for K in [3,5,8,10]:#隐因子
        for steps in [2,100,200]:#迭代次数
            P = np.random.rand(N,K)
            Q = np.random.rand(M,K)
            SI = np.random.rand(M,1) #每个知识点的偏置
            #SVD分解
            nP, nQ, nSI = svdp.matrix_factorization(R, P, Q, SU, SI, K, steps)
            #nR = np.dot(nP, nQ.T)  
            #RMSE评估   
            squaredError = []
            for index, row in test.iterrows(): # 获取每行的值
                #pRating=nR[int(row['csr'])][int(row['ke'])] #获取预测值
                i = int(row['csr'])
                j = int(row['ke'])
                pRating=np.dot(nP[i,:]+SU[i]+nSI[j,0],nQ.T[:,j])
                TRating=row['rating']
                error=TRating-pRating
                squaredError.append(error * error)
            RMSE =sqrt(sum(squaredError) / len(squaredError))#均方根误差RMSE 
            print ("%3d%20d%20.6f" % (K,steps,RMSE))
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))