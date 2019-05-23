# -*- coding: utf-8 -*-
'''
Created on 2017年12月12日
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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
    #2.1：训练集和测试集合并歌曲
    train = train.merge(songs, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    train['song_length'].fillna(200000,inplace=True)
    test['song_length'].fillna(200000,inplace=True)
    #2.2：训练集和测试集合并歌者
    members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)
    members = members.drop(['registration_init_time','expiration_date'], axis=1)
    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')
    #2.3:训练集合测试集合并歌曲信息
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
    songs_ex.drop(['isrc','name'], axis = 1, inplace = True)#song_ex中的name暂不处理
    train = train.merge(songs_ex, on = 'song_id', how = 'left')
    test = test.merge(songs_ex, on = 'song_id', how = 'left')
    del members,songs,songs_ex; gc.collect();
    #2.4：特征提取-统计类特征
    
    
    #2.5：特征提取-信息类特征（爬虫）
    
    ######################################################################
    #3:训练user对物品特征的偏好 song_area\song_year\language\genre_ids\song_length\artist_name\composer\lyricist
    reader = surprise.Reader(rating_scale=(0,1))
    #3.1：msno<->song_area
    mtrain=train[['msno','song_area','target']].dropna().drop_duplicates()#去空值去重
    algo_area=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_area.train(data.build_full_trainset())
    print ('完成msno<->song_area训练')
    #3.2：msno<->song_year
    mtrain=train[['msno','song_year','target']].dropna().drop_duplicates()#去空值去重
    algo_year=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_year.train(data.build_full_trainset())
    print ('完成msno<->song_year训练')
    #3.3：msno<->language
    mtrain=train[['msno','language','target']].dropna().drop_duplicates()#去空值去重
    algo_lang=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_lang.train(data.build_full_trainset())
    print ('完成msno<->language训练')
    #3.4：msno<->song_length
    mlen=train[['msno','song_length','target']].dropna().drop_duplicates()#去空值去重
    _mean_song_length = np.mean(train['song_length'])
    mlen['song_length']=mlen['song_length'].apply(lambda x: 1 if x<=_mean_song_length else 2)#长度转换
    mlen=mlen.drop_duplicates()
    algo_len=surprise.SVD()
    data = surprise.Dataset.load_from_df(mlen,reader)
    algo_len.train(data.build_full_trainset())
    print ('完成msno<->song_length训练')
    #3.6：msno<->genre_ids
    mtrain=train[['msno','genre_ids','target']].dropna()#去空值
    mtrain=mtrain.drop('genre_ids', axis=1).join(mtrain['genre_ids'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genre_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_genre = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','genre_tag','target']],reader)
    algo_genre.train(data.build_full_trainset()) 
    print ('完成msno<->genre_ids训练')
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
    print ('完成msno<->artist_name训练')
    #3.7：msno<->composer
    mtrain=train[['msno','composer','target']].dropna()#去空值
    mtrain['composer']=mtrain['composer'].apply(funsplit)
    mtrain=mtrain.drop('composer', axis=1).join(mtrain['composer'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('composer_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_composer = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','composer_tag','target']],reader)
    algo_composer.train(data.build_full_trainset()) 
    print ('完成msno<->composer训练')
    #3.8：msno<->lyricist
    mtrain=train[['msno','lyricist','target']].dropna()#去空值
    mtrain['lyricist']=mtrain['lyricist'].apply(funsplit)
    mtrain=mtrain.drop('lyricist', axis=1).join(mtrain['lyricist'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('lyricist_tag'))
    mtrain=mtrain.drop_duplicates()#去重
    algo_lyricist = surprise.SVD()#训练
    data = surprise.Dataset.load_from_df(mtrain[['msno','lyricist_tag','target']],reader)
    algo_lyricist.train(data.build_full_trainset()) 
    print ('完成msno<->lyricist训练')
    gc.collect()
    ########################################################################################
    #4：用户对kkbox APP特征的偏好 source_system_tab\source_screen_name\source_type
    #4.1：msno<->source_system_tab
    mtrain=train[['msno','source_system_tab','target']].dropna().drop_duplicates()#去空值去重
    algo_source_system_tab=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_system_tab.train(data.build_full_trainset())
    print ('完成msno<->source_system_tab训练')
    #4.2：msno<->source_screen_name
    mtrain=train[['msno','source_screen_name','target']].dropna().drop_duplicates()#去空值去重
    algo_source_screen_name=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_screen_name.train(data.build_full_trainset())
    print ('完成msno<->source_screen_name训练')
    #4.3：msno<->source_type
    mtrain=train[['msno','source_type','target']].dropna().drop_duplicates()#去空值去重
    algo_source_type=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_source_type.train(data.build_full_trainset())
    print ('完成msno<->source_type训练')
    ########################################################################################
    #5：训练物品为用户特征所偏好 city\bd\gender\registered_via\membership_days
    #5.1：city<->song_id
    mtrain=train[['city','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_city=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_city.train(data.build_full_trainset())
    print ('完成city<->song_id训练')
    #5.2：bd<->song_id
    mtrain=train[['bd','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_bd=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_bd.train(data.build_full_trainset())
    print ('完成bd<->song_id训练')
    #5.3：gender<->song_id
    mtrain=train[['gender','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_gender=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_gender.train(data.build_full_trainset())
    print ('完成gender<->song_id训练')
    #5.4：registered_via<->song_id
    mtrain=train[['registered_via','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_registered_via=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_registered_via.train(data.build_full_trainset())
    print ('完成gregistered_via<->song_id训练')
    #5.5：registered_via<->song_id
    mtrain=train[['membership_days','song_id','target']].dropna().drop_duplicates()#去空值去重
    algo_membership_days=surprise.SVD()
    data = surprise.Dataset.load_from_df(mtrain,reader)
    algo_membership_days.train(data.build_full_trainset())
    print ('完成membership_days<->song_id训练')
    ########################################################################################
    #6：标签SVD后生成的训练集和测试集
    #6.1：训练集
    train['source_system_tab']=train.apply(lambda x:'1:'+str(algo_source_system_tab.predict(x['msno'],x['source_system_tab']).est),axis=1)
    train['source_screen_name']=train.apply(lambda x:'2:'+str(algo_source_screen_name.predict(x['msno'],x['source_screen_name']).est),axis=1)
    train['source_type']=train.apply(lambda x:'3:'+str(algo_source_type.predict(x['msno'],x['source_type']).est),axis=1)
    train['song_area']=train.apply(lambda x:'4:'+str(algo_area.predict(x['msno'],x['song_area']).est),axis=1)
    train['song_year']=train.apply(lambda x:'5:'+str(algo_year.predict(x['msno'],x['song_year']).est),axis=1)
    train['language']=train.apply(lambda x:'6:'+str(algo_lang.predict(x['msno'],x['language']).est),axis=1)
    def songlengthDiscretization(x):
        if x['song_length']<=_mean_song_length:
            return algo_len.predict(x['msno'],1).est
        else:
            return algo_len.predict(x['msno'],2).est
    train['song_length']=train.apply(lambda x:'7:'+str(songlengthDiscretization(x)),axis=1)
    def genreidsGetMax(x):
        avgv=0.0
        num=0
        for tag in str(x['genre_ids']).split('|'):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    train['genre_ids']=train.apply(lambda x:'8:'+str(genreidsGetMax(x)),axis=1)
    def artistnameGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['artist_name'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    train['artist_name']=train.apply(lambda x:'9:'+str(artistnameGetMax(x)),axis=1)
    def composerGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['composer'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    train['composer']=train.apply(lambda x:'10:'+str(composerGetMax(x)),axis=1)
    def lyricistGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['lyricist'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    train['lyricist']=train.apply(lambda x:'11:'+str(lyricistGetMax(x)),axis=1)    
    train['city']=train.apply(lambda x:'1:'+str(algo_city.predict(x['city'],x['song_id']).est),axis=1)    
    train['bd']=train.apply(lambda x:'2:'+str(algo_bd.predict(x['bd'],x['song_id']).est),axis=1)     
    train['gender']=train.apply(lambda x:'3:'+str(algo_gender.predict(x['gender'],x['song_id']).est),axis=1)
    train['registered_via']=train.apply(lambda x:'4:'+str(algo_registered_via.predict(x['registered_via'],x['song_id']).est),axis=1)
    train['membership_days']=train.apply(lambda x:'5:'+str(algo_membership_days.predict(x['membership_days'],x['song_id']).est),axis=1)
    train['gnum']=0#全局特征数
    train['unum']=5#用户特征数
    train['inum']=11#物品特征数
    train=train[['target','gnum','unum','inum','city','bd','gender','registered_via','membership_days','source_system_tab','source_screen_name','source_type','song_area','song_year','language','song_length','genre_ids','artist_name','composer','lyricist']]
    #6.2：测试集   
    test['source_system_tab']=test.apply(lambda x:'1:'+str(algo_source_system_tab.predict(x['msno'],x['source_system_tab']).est),axis=1)
    test['source_screen_name']=test.apply(lambda x:'2:'+str(algo_source_screen_name.predict(x['msno'],x['source_screen_name']).est),axis=1)
    test['source_type']=test.apply(lambda x:'3:'+str(algo_source_type.predict(x['msno'],x['source_type']).est),axis=1)
    test['song_area']=test.apply(lambda x:'4:'+str(algo_area.predict(x['msno'],x['song_area']).est),axis=1)
    test['song_year']=test.apply(lambda x:'5:'+str(algo_year.predict(x['msno'],x['song_year']).est),axis=1)
    test['language']=test.apply(lambda x:'6:'+str(algo_lang.predict(x['msno'],x['language']).est),axis=1)
    def songlengthDiscretization(x):
        if x['song_length']<=_mean_song_length:
            return algo_len.predict(x['msno'],1).est
        else:
            return algo_len.predict(x['msno'],2).est
    test['song_length']=test.apply(lambda x:'7:'+str(songlengthDiscretization(x)),axis=1)
    def genreidsGetMax(x):
        avgv=0.0
        num=0
        for tag in str(x['genre_ids']).split('|'):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    test['genre_ids']=test.apply(lambda x:'8:'+str(genreidsGetMax(x)),axis=1)
    def artistnameGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['artist_name'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    test['artist_name']=test.apply(lambda x:'9:'+str(artistnameGetMax(x)),axis=1)
    def composerGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['composer'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    test['composer']=test.apply(lambda x:'10:'+str(composerGetMax(x)),axis=1)
    def lyricistGetMax(x):
        avgv=0.0
        num=0
        for tag in re.split('/|\\\|;|\||&',str(x['lyricist'])):
            est = algo_genre.predict(x['msno'],tag).est
            avgv=avgv+est
            num=num+1
        return avgv/num
    test['lyricist']=test.apply(lambda x:'11:'+str(lyricistGetMax(x)),axis=1)    
    test['city']=test.apply(lambda x:'1:'+str(algo_city.predict(x['city'],x['song_id']).est),axis=1)    
    test['bd']=test.apply(lambda x:'2:'+str(algo_bd.predict(x['bd'],x['song_id']).est),axis=1)     
    test['gender']=test.apply(lambda x:'3:'+str(algo_gender.predict(x['gender'],x['song_id']).est),axis=1)
    test['registered_via']=test.apply(lambda x:'4:'+str(algo_registered_via.predict(x['registered_via'],x['song_id']).est),axis=1)
    test['membership_days']=test.apply(lambda x:'5:'+str(algo_membership_days.predict(x['membership_days'],x['song_id']).est),axis=1)
    test['target']=0
    test['gnum']=0#全局特征数
    test['unum']=5#用户特征数
    test['inum']=11#物品特征数
    test=test[['target','gnum','unum','inum','city','bd','gender','registered_via','membership_days','source_system_tab','source_screen_name','source_type','song_area','song_year','language','song_length','genre_ids','artist_name','composer','lyricist']]
    ################################################################################################
    train.to_csv(homedir + '/ua.base.example', index=False,encoding='utf8',sep=' ',header=False)
    test.to_csv(homedir + '/ua.test.example', index=False,encoding='utf8',sep=' ',header=False)
    '''
    #7：lgbm模型训练
    #7.1：训练集划分
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
    #7.2：lgbm训练
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
    #8:输出预测结果
    predictions = lgbm_model.predict(X_test)
    # Writing output to file
    subm = pd.DataFrame()
    subm['id'] = ids
    subm['target'] = predictions
    #subm=subm.groupby('id', as_index=False)['target'].max()
    subm.to_csv(homedir + '/svd_lgbm_sub.csv', index=False, float_format = '%.5f')
    '''
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))       
