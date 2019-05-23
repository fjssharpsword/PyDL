# -*- coding: utf-8 -*-
'''
Created on 2017年11月29日
@author: Jason.F_CN
@summary: Content-Based Recommendations，CBR
'''
import time
import pandas as pd
import os
import numpy as np
import re
###################################################################

if __name__ == "__main__":   
    start = time.clock()   
    
    #reload(sys)
    #sys.setdefaultencoding('utf8')
    homedir = os.getcwd()#获取当前文件的路径
    #1：数据导入
    #1.1：训练集
    homedir = os.getcwd()#获取当前文件的路径 
    #1.1：训练集
    train=pd.read_csv(homedir+'/train.csv',dtype={'msno' : 'str','source_system_tab' : 'str','source_screen_name' : 'str','source_type' : 'str','target' : np.uint8,'song_id' : 'str'})
    print '训练集，有：', train.shape[0], '行', train.shape[1], '列'
    #1.2：测试集
    test=pd.read_csv(homedir+'/test.csv',dtype={'msno' : 'str','source_system_tab' : 'str','source_screen_name' : 'str','source_type' : 'str','song_id' : 'str'})
    print '测试集，有：', test.shape[0], '行', test.shape[1], '列'
    #1.3：听者
    members=pd.read_csv(homedir+'/members.csv',dtype={'city' : 'str','bd' : 'str','gender' : 'str','registered_via' : 'str'},parse_dates=['registration_init_time','expiration_date'])
    print '听歌者，有：', members.shape[0], '行', members.shape[1], '列'
    #1.4：乐曲
    songs=pd.read_csv(homedir+'/songs.csv',dtype={'genre_ids': 'str','language' : 'str','artist_name' : 'str','composer' : 'str', 'lyricist' : 'str','song_id' : 'str','song_length':np.uint16})
    print '歌曲集，有：', songs.shape[0], '行', songs.shape[1], '列'  
    songs_ex=pd.read_csv(homedir+'/song_extra_info.csv')
    print '歌曲信息集，有：', songs_ex.shape[0], '行', songs_ex.shape[1], '列' 
    ###########################################################################
    #2：数据集合并
    #1.1：训练集和测试集合并歌曲
    train = train.merge(songs, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    #1.2：训练集和测试集合并歌者
    members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)
    members = members.drop(['registration_init_time','expiration_date'], axis=1)
    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')
    #1.3:训练集合测试集合并歌曲信息
    def isrc_to_year(isrc):
        if type(isrc) == str:
            if int(isrc[5:7]) > 17:
                return 1900 + int(isrc[5:7])
            else:
                return 2000 + int(isrc[5:7])
        else:
            return np.nan       
    def isrc_to_area(isrc):
        if type(isrc) == str:
            return (isrc[0:2])
        else:
            return np.nan   
    songs_ex['song_year'] = songs_ex['isrc'].apply(isrc_to_year)
    songs_ex['song_area'] = songs_ex['isrc'].apply(isrc_to_area)
    songs_ex.drop(['isrc', 'name'], axis = 1, inplace = True)#song_ex中的name暂不处理
    train = train.merge(songs_ex, on = 'song_id', how = 'left')
    test = test.merge(songs_ex, on = 'song_id', how = 'left')
    train['song_length'].fillna(0,inplace=True)
    test['song_length'].fillna(0,inplace=True)
    train.fillna('no',inplace=True)
    test.fillna('no',inplace=True)
    ###########################################################################
    #3：余弦相似度
    submit = []
    for index,row in test.iterrows():
        iid=row['id']
        mid=row['msno']
        sid=row['song_id']
        #1:从train中找出对应mid和sid的子训练集
        sub_train=pd.concat([train.loc[(train['msno']==mid)],train.loc[(train['song_id']==sid)]],axis=0).drop_duplicates()
        #2:计算子训练集和该预测条的cos相似度
        target=0.0
        for sin,srow in sub_train.iterrows():
            count=0.0
            if (row['source_system_tab']==srow['source_system_tab'] and row['source_system_tab']!='no' and srow['source_system_tab']!='no'): 
                count=count+1
            if (row['source_screen_name']==srow['source_screen_name'] and row['source_screen_name']!='no' and srow['source_screen_name']!='no'): 
                count=count+1
            if (row['source_type']==srow['source_type'] and row['source_type']!='no' and srow['source_type']!='no'): count=count+1
            if (abs(row['song_length']-srow['song_length'])<30000 and row['song_length']!=0 and srow['song_length']!=0): 
                count=count+1  #音乐长度小于30s就算相似
            listA=re.split('/|\\\|;|\||&',row['genre_ids'])
            listB=re.split('/|\\\|;|\||&',srow['genre_ids'])
            retA = [i for i in listA if i in listB] #两个list的交集
            if len(retA)>0 :count=count+1 #说明有一个体裁一样
            listA=re.split('/|\\\|;|\||&',row['artist_name'])
            listB=re.split('/|\\\|;|\||&',srow['artist_name'])
            retA = [i for i in listA if i in listB] #两个list的交集
            if len(retA)>0 :count=count+1 #说明有一个体裁一样
            listA=re.split('/|\\\|;|\||&',row['composer'])
            listB=re.split('/|\\\|;|\||&',srow['composer'])
            retA = [i for i in listA if i in listB] #两个list的交集
            if len(retA)>0 :count=count+1 #说明有一个体裁一样
            listA=re.split('/|\\\|;|\||&',row['lyricist'])
            listB=re.split('/|\\\|;|\||&',srow['lyricist'])
            retA = [i for i in listA if i in listB] #两个list的交集
            if len(retA)>0 :count=count+1 #说明有一个体裁一样
            if (row['language']==srow['language'] and row['language']!='no' and srow['language']!='no'): 
                count=count+1
            if (row['city']==srow['city'] and row['city']!='no' and srow['city']!='no'): 
                count=count+1
            if (row['bd']==srow['bd'] and row['bd']!='no' and srow['bd']!='no'): 
                count=count+1
            if (row['gender']==srow['gender'] and row['gender']!='no' and srow['gender']!='no'): 
                count=count+1
            if (row['registered_via']==srow['registered_via'] and row['registered_via']!='no' and srow['registered_via']!='no'): 
                count=count+1
            if (row['membership_days']==srow['membership_days'] and row['membership_days']!='no' and srow['membership_days']!='no'): 
                count=count+1
            if (row['song_year']==srow['song_year'] and row['song_year']!='no' and srow['song_year']!='no'): 
                count=count+1
            if (row['song_area']==srow['song_area'] and row['song_area']!='no' and srow['song_area']!='no'): 
                count=count+1
            if (target<(count/16)): target=count/16
        submit.append((iid,target))
        print (str(iid) +':' +str(target))
    #输出预测结果
    submit = pd.DataFrame(submit,columns=['id','target'])
    submit.to_csv(homedir+'/cos_sub.csv',index=False,encoding='utf8')    
        
    
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))       
