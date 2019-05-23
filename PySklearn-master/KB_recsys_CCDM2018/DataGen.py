# -*- coding: utf-8 -*-
'''
Created on 2018年3月5日

@author: Jason.F
@summary: CCDM2018 知识项推荐-生成csr-ke-num数据集
'''
import time
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np 

if __name__ == "__main__":    
    start = time.clock() 
    
    homedir = os.getcwd()#获取当前文件的路径
    df_kb = pd.read_csv(homedir+'/kb1.csv',names=['no','csr','city','ke','kepath','time'],header=0,encoding='gbk')
    df=df_kb[['csr','ke']]
    df_kb = pd.read_csv(homedir+'/kb2.csv',names=['no','csr','city','ke','kepath','time'],header=0,encoding='gbk')
    df=df.append(df_kb[['csr','ke']])
    df_kb = pd.read_csv(homedir+'/kb3.csv',names=['no','csr','city','ke','kepath','time'],header=0,encoding='gbk')
    df=df.append(df_kb[['csr','ke']])
    del df_kb
    le = LabelEncoder()
    df_encoded = df.apply(le.fit_transform)#将csr和ke全部标准编号
    del df
    #print (df_encoded.groupby('csr').size())#12320
    #print (df_encoded.groupby('ke').size())#14559
    dict_num={}
    for csr,ke in  np.array(df_encoded).tolist():
        dict_num.setdefault(csr,{})
        dict_num[csr].setdefault(ke,0)
        dict_num[csr][ke] += 1
    del df_encoded
    list_num=[]
    for csr,v in dict_num.items():
        for ke,num in v.items():
            list_num.append([csr,ke,num])
    del dict_num    
    #df=pd.DataFrame(dict_num.items(), columns=['csr', 'ke','num'])#dict转化为 dataframe
    df=pd.DataFrame(list_num, columns=['csr', 'ke','num'])#list转换成 dataframe
    df.to_csv(homedir+'/kb.csv',index=None)
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))
