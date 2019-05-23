# -*- coding: utf-8 -*-
'''
Created on 2017年11月22日
@author: Jason.F_CN
@summary: LGBM
'''
import time
import pandas as pd
import os
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import re
#####################################################
if __name__ == "__main__":   
    start = time.clock()   
    
    #1：数据导入
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
    #2：数据集合并
    #1.1：训练集和测试集合并歌曲
    train = train.merge(songs, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    #1.2：训练集和测试集合并歌者
    members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)
    members['registration_year'] = members['registration_init_time'].dt.year
    members['registration_month'] = members['registration_init_time'].dt.month
    members['registration_date'] = members['registration_init_time'].dt.day
    members['expiration_year'] = members['expiration_date'].dt.year
    members['expiration_month'] = members['expiration_date'].dt.month
    members['expiration_date'] = members['expiration_date'].dt.day
    members = members.drop(['registration_init_time'], axis=1)
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
    songs_ex.drop(['isrc', 'name'], axis = 1, inplace = True)
    train = train.merge(songs_ex, on = 'song_id', how = 'left')
    test = test.merge(songs_ex, on = 'song_id', how = 'left')
    #1.4：大训练集合和大测试集处理
    train.song_length.fillna(200000,inplace=True)
    train.song_length = train.song_length.astype(np.uint32)
    train.song_id = train.song_id.astype('category')
    test.song_length.fillna(200000,inplace=True)
    test.song_length = test.song_length.astype(np.uint32)
    test.song_id = test.song_id.astype('category')
    #Converting object types to categorical
    train = pd.concat([
        train.select_dtypes([], ['object']),
        train.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex_axis(train.columns, axis=1)

    test = pd.concat([
        test.select_dtypes([], ['object']),
        test.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex_axis(test.columns, axis=1)
    #1.5：回收资源
    del members,songs,songs_ex; gc.collect();
    ################################################################################################
    #4：特征处理
    #4.1：作词者处理
    def lyricist_count(x):
        if x == 'no_lyricist':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
        return sum(map(x.count, ['|', '/', '\\', ';']))

    train['lyricist'] = train['lyricist'].cat.add_categories(['no_lyricist'])
    train['lyricist'].fillna('no_lyricist',inplace=True)
    train['lyricists_count'] = train['lyricist'].apply(lyricist_count).astype(np.int8)
    test['lyricist'] = test['lyricist'].cat.add_categories(['no_lyricist'])
    test['lyricist'].fillna('no_lyricist',inplace=True)
    test['lyricists_count'] = test['lyricist'].apply(lyricist_count).astype(np.int8)
    #4.2：作曲者处理
    def composer_count(x):
        if x == 'no_composer':
            return 0
        else:
            return sum(map(x.count, ['|', '/', '\\', ';'])) + 1
    train['composer'] = train['composer'].cat.add_categories(['no_composer'])
    train['composer'].fillna('no_composer',inplace=True)
    train['composer_count'] = train['composer'].apply(composer_count).astype(np.int8)
    test['composer'] = test['composer'].cat.add_categories(['no_composer'])
    test['composer'].fillna('no_composer',inplace=True)
    test['composer_count'] = test['composer'].apply(composer_count).astype(np.int8)
    #4.3：艺术家处理
    def is_featured(x):
        if 'feat' in str(x) :
            return 1
        return 0
    train['artist_name'] = train['artist_name'].cat.add_categories(['no_artist'])
    train['artist_name'].fillna('no_artist',inplace=True)
    train['is_featured'] = train['artist_name'].apply(is_featured).astype(np.int8)
    test['artist_name'] = test['artist_name'].cat.add_categories(['no_artist'])
    test['artist_name'].fillna('no_artist',inplace=True)
    test['is_featured'] = test['artist_name'].apply(is_featured).astype(np.int8)
    def artist_count(x):
        if x == 'no_artist' or x=='佚名' or x=='群星':
            return 0
        else:
            return x.count('and') + x.count(',') + x.count('feat') + x.count('&')
    train['artist_count'] = train['artist_name'].apply(artist_count).astype(np.int8)
    test['artist_count'] = test['artist_name'].apply(artist_count).astype(np.int8)
    # if artist is same as composer
    train['artist_composer'] = (np.asarray(train['artist_name']) == np.asarray(train['composer'])).astype(np.int8)
    test['artist_composer'] = (np.asarray(test['artist_name']) == np.asarray(test['composer'])).astype(np.int8)
    # if artist, lyricist and composer are all three same
    train['artist_composer_lyricist'] = ((np.asarray(train['artist_name']) == np.asarray(train['composer'])) & 
                                     np.asarray((train['artist_name']) == np.asarray(train['lyricist'])) & 
                                     np.asarray((train['composer']) == np.asarray(train['lyricist']))).astype(np.int8)
    test['artist_composer_lyricist'] = ((np.asarray(test['artist_name']) == np.asarray(test['composer'])) & 
                                    (np.asarray(test['artist_name']) == np.asarray(test['lyricist'])) &
                                    np.asarray((test['composer']) == np.asarray(test['lyricist']))).astype(np.int8)
    #4.4：语言处理
    def song_lang_boolean(x):
        if '17.0' in str(x) or '45.0' in str(x):
            return 1
        return 0
    train['song_lang_boolean'] = train['language'].apply(song_lang_boolean).astype(np.int8)
    test['song_lang_boolean'] = test['language'].apply(song_lang_boolean).astype(np.int8)                                
    #4.5：长度处理
    _mean_song_length = np.mean(train['song_length'])
    def smaller_song(x):
        if x < _mean_song_length:
            return 1
        return 0
    train['smaller_song'] = train['song_length'].apply(smaller_song).astype(np.int8)
    test['smaller_song'] = test['song_length'].apply(smaller_song).astype(np.int8)
    #4.6：统计歌曲播放次数
    _dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
    _dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}
    def count_song_played(x):
        try:
            return _dict_count_song_played_train[x]
        except KeyError:
            try:
                return _dict_count_song_played_test[x]
            except KeyError:
                return 0
    train['count_song_played'] = train['song_id'].apply(count_song_played).astype(np.int64)
    test['count_song_played'] = test['song_id'].apply(count_song_played).astype(np.int64)
    #4.7：统计艺术家被听次数
    _dict_count_artist_played_train = {k: v for k, v in train['artist_name'].value_counts().iteritems()}
    _dict_count_artist_played_test = {k: v for k, v in test['artist_name'].value_counts().iteritems()}
    def count_artist_played(x):
        try:
            return _dict_count_artist_played_train[x]
        except KeyError:
            try:
                return _dict_count_artist_played_test[x]
            except KeyError:
                return 0
    train['count_artist_played'] = train['artist_name'].apply(count_artist_played).astype(np.int64)
    test['count_artist_played'] = test['artist_name'].apply(count_artist_played).astype(np.int64) 
    #4.8：lyricist/artist_name/composer/genre_ids拆分成多行
    gc.collect();
    def fun_strsplit(x):
        sstr=''
        for s in re.split('/|\\\|;|\||&',x):
            sstr=sstr+'|'+s
        l=len(sstr)
        return sstr[1:l]
    train['artist_name']=train['artist_name'].apply(fun_strsplit)
    train=train.drop('artist_name', axis=1).join(train['artist_name'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('artist_tag'))
    #train['lyricist']=train['lyricist'].apply(fun_strsplit)
    #train=train.drop('lyricist', axis=1).join(train['lyricist'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('lyricist_tag'))
    #train['composer']=train['composer'].apply(fun_strsplit)
    #train=train.drop('composer', axis=1).join(train['composer'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('composer_tag'))
    train['genre_ids']=train['genre_ids'].apply(fun_strsplit)
    train=train.drop('genre_ids', axis=1).join(train['genre_ids'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genre_tag'))
    test['artist_name']=test['artist_name'].apply(fun_strsplit)
    test=test.drop('artist_name', axis=1).join(test['artist_name'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('artist_tag'))
    #test['lyricist']=test['lyricist'].apply(fun_strsplit)
    #test=test.drop('lyricist', axis=1).join(test['lyricist'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('lyricist_tag'))
    #test['composer']=test['composer'].apply(fun_strsplit)
    #test=test.drop('composer', axis=1).join(test['composer'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('composer_tag'))
    test['genre_ids']=test['genre_ids'].apply(fun_strsplit)
    test=test.drop('genre_ids', axis=1).join(test['genre_ids'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genre_tag'))
    gc.collect();
    ################################################################################################
    #5：lgbm模型训练
    #5.1：训练集划分
    for col in train.columns:
        if train[col].dtype == object:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
    X_train = train.drop(['target'], axis=1)
    y_train = train['target'].values
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)
    X_test = test.drop(['id'], axis=1)
    ids = test['id'].values
    del train, test; gc.collect();#回收资源
    #5.2：lgbm训练
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val)
    params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.2 ,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 100,
        'metric' : 'auc'
    }
    lgbm_model = lgb.train(params, train_set = lgb_train, valid_sets = lgb_val, verbose_eval=5)
    ##############################################################################################
    #6:输出预测结果
    predictions = lgbm_model.predict(X_test)
    # Writing output to file
    subm = pd.DataFrame()
    subm['id'] = ids
    subm['target'] = predictions
    subm=subm.groupby('id', as_index=False)['target'].max()
    subm.to_csv(homedir + '/lgbm_sub.csv', index=False, float_format = '%.5f')
    
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))       
