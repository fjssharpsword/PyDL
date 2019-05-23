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
import gc
import surprise
from surprise import accuracy
import re
import sys
###################################################################

if __name__ == "__main__":   
    start = time.clock()   
    
    reload(sys)
    sys.setdefaultencoding('utf8')
    homedir = os.getcwd()#获取当前文件的路径
    #1：数据导入
    #1.1：训练集
    homedir = os.getcwd()#获取当前文件的路径 
    #1.1：训练集
    train=pd.read_csv(homedir+'/train.csv',dtype={'msno' : 'category','source_system_tab' : 'category','source_screen_name' : 'category','source_type' : 'category','target' : np.uint8,'song_id' : 'category'})
    print '训练集，有：', train.shape[0], '行', train.shape[1], '列'
    #1.2：测试集
    test=pd.read_csv(homedir+'/test.csv',dtype={'msno' : 'category','source_system_tab' : 'category','source_screen_name' : 'category','source_type' : 'category','song_id' : 'category'})
    print '测试集，有：', test.shape[0], '行', test.shape[1], '列'
    #1.3：听者
    members=pd.read_csv(homedir+'/members.csv',dtype={'city' : 'category','bd' : np.uint8,'gender' : 'category','registered_via' : 'category'},parse_dates=['registration_init_time','expiration_date'])
    print '听歌者，有：', members.shape[0], '行', members.shape[1], '列'
    #1.4：乐曲
    songs=pd.read_csv(homedir+'/songs.csv',dtype={'genre_ids': 'category','language' : 'category','artist_name' : 'category','composer' : 'category', 'lyricist' : 'category','song_id' : 'category'})
    print '歌曲集，有：', songs.shape[0], '行', songs.shape[1], '列'  
    songs_ex=pd.read_csv(homedir+'/song_extra_info.csv')
    print '歌曲信息集，有：', songs_ex.shape[0], '行', songs_ex.shape[1], '列' 
    ###########################################################################
    #2：基础特征处理
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
    songs_ex.drop(['isrc'], axis = 1, inplace = True)#song_ex中的name暂不处理
    train = train.merge(songs_ex, on = 'song_id', how = 'left')
    test = test.merge(songs_ex, on = 'song_id', how = 'left')
    del members,songs,songs_ex; gc.collect();
    ######################################################################
    #3:训练user对物品特征的偏好 song_area\song_year\language\genre_ids\song_length\artist_name\composer\lyricist
    reader = surprise.Reader(rating_scale=(0,1))
    #3.1：msno<->song_area
    mtrain=train[['msno','song_area','target']].dropna().drop_duplicates()#去空值去重
    algo_area=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_area.train(data.build_full_trainset())
    predictions = algo_area.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->song_area训练')
    #3.2：msno<->song_year
    mtrain=train[['msno','song_year','target']].dropna().drop_duplicates()#去空值去重
    algo_year=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_year.train(data.build_full_trainset())
    predictions = algo_year.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->song_year训练')
    #3.3：msno<->language
    mtrain=train[['msno','language','target']].dropna().drop_duplicates()#去空值去重
    algo_lang=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_lang.train(data.build_full_trainset())
    predictions = algo_lang.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->language训练')
    #3.4：msno<->song_length
    mlen=train[['msno','song_length','target']].dropna().drop_duplicates()#去空值去重
    song_length_avg=mlen['song_length'].sum()/mlen['song_length'].count()
    mlen['song_length'].fillna(song_length_avg,inplace=True)
    mlen['song_length']=mlen['song_length'].apply(lambda x: 0 if x<=song_length_avg else 1)#长度转换
    mlen=mlen.drop_duplicates()
    algo_len=surprise.SVD()
    data = surprise.Dataset.load_from_df(mlen,reader)
    algo_len.train(data.build_full_trainset())
    predictions = algo_len.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->song_length训练')
    #3.6：msno<->genre_ids
    mtrain=train[['msno','genre_ids','target']].dropna()#去空值
    mtrain=mtrain.drop('genre_ids', axis=1).join(mtrain['genre_ids'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genre_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_genre = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','genre_tag','target']],reader)
    algo_genre.train(data.build_full_trainset()) 
    predictions = algo_genre.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True)
    print ('完成msno<->genre_ids训练')
    gc.collect()
    #3.6：msno<->artist_name
    def funsplit(x):
        sstr=''
        for s in re.split('/|\\\|;|\||&',x):
            sstr=sstr+'|'+s
        l=len(sstr)
        return sstr[1:l]
    mtrain=train[['msno','artist_name','target']].dropna()#去空值
    mtrain['artist_name']=mtrain['artist_name'].apply(funsplit)
    mtrain=mtrain.drop('artist_name', axis=1).join(mtrain['artist_name'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('artist_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_artist = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','artist_tag','target']],reader)
    algo_artist.train(data.build_full_trainset()) 
    predictions = algo_artist.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True)
    print ('完成msno<->artist_name')
    gc.collect()
    #3.7：msno<->composer
    mtrain=train[['msno','composer','target']].dropna()#去空值
    mtrain['composer']=mtrain['composer'].apply(funsplit)
    mtrain=mtrain.drop('composer', axis=1).join(mtrain['composer'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('composer_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_composer = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','composer_tag','target']],reader)
    algo_composer.train(data.build_full_trainset()) 
    predictions = algo_composer.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True)
    print ('完成msno<->composer')
    gc.collect()
    #3.8：msno<->lyricist
    mtrain=train[['msno','lyricist','target']].dropna()#去空值
    mtrain['lyricist']=mtrain['lyricist'].apply(funsplit)
    mtrain=mtrain.drop('lyricist', axis=1).join(mtrain['lyricist'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('lyricist_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_lyricist = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','lyricist_tag','target']],reader)
    algo_lyricist.train(data.build_full_trainset()) 
    predictions = algo_lyricist.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True)
    print ('完成msno<->lyricist')
    gc.collect()
    ########################################################################################
    #4：用户对kkbox APP特征的偏好 source_system_tab\source_screen_name\source_type
    #4.1：msno<->source_system_tab
    mtrain=train[['msno','source_system_tab','target']].dropna().drop_duplicates()#去空值去重
    algo_source_system_tab=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_system_tab.train(data.build_full_trainset())
    predictions = algo_source_system_tab.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->source_system_tab训练')
    #4.2：msno<->source_screen_name
    mtrain=train[['msno','source_screen_name','target']].dropna().drop_duplicates()#去空值去重
    algo_source_screen_name=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_screen_name.train(data.build_full_trainset())
    predictions = algo_source_screen_name.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->source_screen_name训练')
    #4.3：msno<->source_type
    mtrain=train[['msno','source_type','target']].dropna().drop_duplicates()#去空值去重
    algo_source_type=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_type.train(data.build_full_trainset())
    predictions = algo_source_type.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成msno<->source_type训练')
    ########################################################################################
    #5：训练物品为用户特征所偏好 city\bd\gender\registered_via\membership_days
    #5.1：city<->song_id
    mtrain=train[['city','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_city=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_city.train(data.build_full_trainset())
    predictions = algo_city.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成city<->song_id训练')
    #5.2：bd<->song_id
    mtrain=train[['bd','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_bd=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_bd.train(data.build_full_trainset())
    predictions = algo_bd.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成bd<->song_id训练')
    #5.3：gender<->song_id
    mtrain=train[['gender','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_gender=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_gender.train(data.build_full_trainset())
    predictions = algo_gender.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成gender<->song_id训练')
    #5.4：registered_via<->song_id
    mtrain=train[['registered_via','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_registered_via=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_registered_via.train(data.build_full_trainset())
    predictions = algo_registered_via.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成gregistered_via<->song_id训练')
    #5.5：registered_via<->song_id
    mtrain=train[['membership_days','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_membership_days=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_membership_days.train(data.build_full_trainset())
    predictions = algo_membership_days.test(data.build_full_trainset().build_testset())#测试
    accuracy.rmse(predictions, verbose=True) 
    print ('完成membership_days<->song_id训练')
    ########################################################################################
    #5：预测并提交
    submit = []
    for index,row in test.iterrows():
        est=0.0
        count=0
        #msno<->song_area
        est_tmp=algo_area.predict(row['msno'],row['song_area']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #msno<->song_year
        est_tmp=algo_year.predict(row['msno'],row['song_year']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #msno<->language
        est_tmp=algo_lang.predict(row['msno'],row['language']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #msno<->song_length
        if (row['language']<=song_length_avg):
            est_tmp=algo_len.predict(row['msno'],0).est
        else:
            est_tmp=algo_len.predict(row['msno'],1).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #msno<->genre_ids
        max_genre=0
        genre_tags=str(row['genre_ids']).split('|')
        for genre_tag in genre_tags:
            est_genre = algo_genre.predict(row['msno'],genre_tag).est
            if (est_genre>max_genre):
                max_genre=est_genre
        if (max_genre>0): 
            est=est+max_genre
            count=count+1
        #msno<->artist_name
        max_art=0
        for art_tag in re.split('/|\\\|;|\||&',str(row['artist_name'])):
            est_art = algo_artist.predict(row['msno'],art_tag).est
            if (max_art<est_art):
                max_art=est_art
        if (max_art>0): 
            est=est+max_art
            count=count+1
        #msno<->composer
        max_com=0
        for com_tag in re.split('/|\\\|;|\||&',str(row['composer'])):
            est_com = algo_composer.predict(row['msno'],com_tag).est
            if (max_com<est_com):
                max_com=est_com
        if (max_com>0): 
            est=est+max_com
            count=count+1
        #msno<->lyricist
        max_lyr=0
        for lyr_tag in re.split('/|\\\|;|\||&',str(row['lyricist'])):
            est_lyr = algo_composer.predict(row['msno'],lyr_tag).est
            if (max_lyr<est_lyr):
                max_lyr=est_lyr
        if (max_lyr>0): 
            est=est+max_lyr
            count=count+1
        #city<->song_id
        est_tmp=algo_city.predict(row['city'],row['song_id']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #bd<->song_id
        est_tmp=algo_bd.predict(row['bd'],row['song_id']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #bd<->song_id
        est_tmp=algo_gender.predict(row['gender'],row['song_id']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #registered_via<->song_id
        est_tmp=algo_registered_via.predict(row['registered_via'],row['song_id']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        #membership_days<->song_id
        est_tmp=algo_membership_days.predict(row['membership_days'],row['song_id']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1  
        #msno<->source_system_tab
        est_tmp=algo_source_system_tab.predict(row['msno'],row['source_system_tab']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        est_tmp=algo_source_screen_name.predict(row['msno'],row['source_screen_name']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1
        est_tmp=algo_source_type.predict(row['msno'],row['source_type']).est
        if (est_tmp>0): 
            est=est+est_tmp
            count=count+1    
           
        submit.append((row['id'],est/count))
    #输出预测结果
    submit = pd.DataFrame(submit,columns=['id','target'])
    submit.to_csv(homedir+'/cbr_sub.csv',index=False,encoding='utf8')
    
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))       
