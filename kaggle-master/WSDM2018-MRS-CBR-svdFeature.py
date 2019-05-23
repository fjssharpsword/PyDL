# -*- coding: utf-8 -*-
'''
Created on 2017年12月07日
@author: Jason.F_CN
@summary: SVDFeature 
'''
import time
import pandas as pd
import os
import numpy as np
import re
import gc
###################################################################

if __name__ == "__main__":   
    start = time.clock()   
    
    homedir = os.getcwd()#获取当前文件的路径
    #1：数据导入 
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
    #特征向量化
    #2.1：msno和song_id向量化
    tlist=list(set (train['msno'].values))
    tlist.extend(list(set (test['msno'].values)))
    mid_list=list(set(tlist))
    tlist=list(set (train['song_id'].values))
    tlist.extend(list(set (test['song_id'].values)))
    sid_list=list(set(tlist))
    #2.2：全局特征source_system_tab\source_screen_name\source_type
    #2.2.1:source_system_tab
    tlist=list(set (train['source_system_tab'].values))
    tlist.extend(list(set (test['source_system_tab'].values)))
    gid_sst=list(set(tlist))
    #2.2.2:source_screen_name
    tlist=list(set (train['source_screen_name'].values))
    tlist.extend(list(set (test['source_screen_name'].values)))
    gid_ssn=list(set(tlist))
    #2.2.3:source_type
    tlist=list(set (train['source_type'].values))
    tlist.extend(list(set (test['source_type'].values)))
    gid_st=list(set(tlist))
    del tlist;gc.collect()
    #2.3：member特征
    members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(int)
    members = members.drop(['registration_init_time','expiration_date'], axis=1)
    def gendertovector(x):
        if x=='male':
            return 1
        elif x=='female':
            return 2
        else :
            return 0
    members['gender']=members['gender'].apply(gendertovector)
    train = train.merge(members, on='msno', how='left')
    test = test.merge(members, on='msno', how='left')
    #2.4：song_ex特征
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
    area_list=list(set (songs_ex['song_area'].values))
    songs_ex.drop(['isrc', 'name'], axis = 1, inplace = True)#song_ex中的name暂不处理
    train = train.merge(songs_ex, on = 'song_id', how = 'left')
    train['song_year'].fillna('0',inplace=True)
    test = test.merge(songs_ex, on = 'song_id', how = 'left')
    test['song_year'].fillna('0',inplace=True)
    #2.5：songs特征
    #2.5.1：artist_name
    artist=[]
    songs['artist_name'].fillna('no',inplace=True)
    for slist in list(set (songs['artist_name'].values)):
        for word in re.split('/|\\\|;|\||&',slist):
            word=word.decode('utf-8')
            artist.append(word)
    composer=[]
    songs['composer'].fillna('no',inplace=True)
    for slist in list(set (songs['composer'].values)):
        for word in re.split('/|\\\|;|\||&',slist):
            word=word.decode('utf-8')
            composer.append(word)
    lyricist=[]
    songs['lyricist'].fillna('no',inplace=True)
    for slist in list(set (songs['lyricist'].values)):
        for word in re.split('/|\\\|;|\||&',slist):
            word=word.decode('utf-8')
            lyricist.append(word)
    train = train.merge(songs, on='song_id', how='left')
    train['genre_ids'].fillna('no',inplace=True)
    train['song_length'].fillna('0',inplace=True)
    train['artist_name'].fillna('no',inplace=True)
    train['composer'].fillna('no',inplace=True)
    train['lyricist'].fillna('no',inplace=True)
    test = test.merge(songs, on='song_id', how='left') 
    test['genre_ids'].fillna('no',inplace=True)   
    test['song_length'].fillna('0',inplace=True)
    test['artist_name'].fillna('no',inplace=True)    
    test['composer'].fillna('no',inplace=True)
    test['lyricist'].fillna('no',inplace=True)
    ###########################################################################
    #3:生成SVDFeature的uabase和uatest
    
    #3.1：训练集
    uabase = []
    for index,row in train.iterrows():
        rbase=[]
        target=int(row['target'])#target
        #全局特征
        gid_num=0
        #source_system_tab
        str_sst=row['source_system_tab']
        if (str_sst in gid_sst):
            rbase.append('%d:%d' % (1,gid_sst.index(str_sst)+1))
            gid_num=gid_num+1
        #source_screen_name
        str_ssn=row['source_screen_name']
        if (str_ssn in gid_ssn):
            rbase.append('%d:%d' % (2,gid_ssn.index(str_ssn)+1))
            gid_num=gid_num+1
        #source_type
        str_st=row['source_type']
        if (str_st in gid_st):
            rbase.append('%d:%d' % (3,gid_st.index(str_st)+1))
            gid_num=gid_num+1
        #用户特征
        uid_num=0
        #msno
        rbase.append('%d:%d' % (1,mid_list.index(row['msno'])+1))
        uid_num=uid_num+1
        #city
        city_value=row['city']
        if city_value and int(city_value)!=0:#不为空
            rbase.append('%d:%d' % (2,int(city_value)))
            uid_num=uid_num+1
        #bd
        bd_value=row['bd']
        if bd_value and 0<int(bd_value)<100:#不为空
            rbase.append('%d:%d' % (3,int(bd_value)))
            uid_num=uid_num+1
        #gender
        gender_value=row['gender']
        if gender_value and int(gender_value)!=0:#不为空
            rbase.append('%d:%d' % (4,int(gender_value)))
            uid_num=uid_num+1
        #registered_via
        registered_via_value=row['registered_via']
        if registered_via_value and int(registered_via_value)!=0:#不为空
            rbase.append('%d:%d' % (5,int(registered_via_value)))
            uid_num=uid_num+1
        #membership_days
        membership_days_value=row['membership_days']
        if membership_days_value and int(membership_days_value)!=0:#不为空
            rbase.append('%d:%d' % (6,int(membership_days_value)))
            uid_num=uid_num+1
        #物品特征
        sid_num=0
        #song_id
        rbase.append('%d:%d' % (1,sid_list.index(row['song_id'])+1))
        sid_num=sid_num+1
        #language
        language_value=row['language']
        if language_value and language_value!='0':#不为空
            rbase.append('%d:%s' % (2,language_value))
            sid_num=sid_num+1
        #song_length
        song_length_value=row['song_length']
        if song_length_value:
            if  int(song_length_value)!=0:#不为空
                rbase.append('%d:%d' % (3,int(song_length_value)))
                sid_num=sid_num+1
        #song_year
        song_year_value=row['song_year']
        if song_year_value :
            if  song_year_value!='0':#不为空
                rbase.append('%d:%s' % (4,song_year_value))
                sid_num=sid_num+1
        #song_area
        song_area_value=row['song_area']
        if song_area_value:
            rbase.append('%d:%d' % (5,area_list.index(song_area_value)+1))
            sid_num=sid_num+1       
        #genre_ids
        genre_value=row['genre_ids']
        if genre_value:#不为空
            fgenre=str(genre_value).split('|')[0]
            if fgenre!='no' and fgenre!='nan':
                rbase.append('%d:%s' % (6,fgenre))
                sid_num=sid_num+1
        #artist_name
        artist_value=row['artist_name']
        if artist_value!='no':
            fartist=re.split('/|\\\|;|\||&',artist_value)[0]
            if fartist in artist:
                rbase.append('%d:%d' % (7,artist.index(fartist)+1))
                sid_num=sid_num+1          
        #composer
        composer_value=row['composer']
        if composer_value!='no':
            fcomposer=re.split('/|\\\|;|\||&',composer_value)[0]
            if fcomposer in composer:
                rbase.append('%d:%d' % (8,composer.index(fcomposer)+1))
                sid_num=sid_num+1           
        #lyricist
        lyricist_value=row['lyricist']
        if lyricist_value!='no':
            flyricist=re.split('/|\\\|;|\||&',lyricist_value)[0]
            if flyricist in lyricist:
                rbase.append('%d:%d' % (9,lyricist.index(flyricist)+1))
                sid_num=sid_num+1      
        #特征数插入
        rbase.insert(0,sid_num)
        rbase.insert(0,uid_num)
        rbase.insert(0,gid_num)
        rbase.insert(0,target)
        print (rbase)
        uabase.append(rbase)
    uabase.to_csv(homedir + '/ua.base', index=False,encoding='utf8',sep=' ',header=False)
    
    #3.2：预测集
    uatest = []
    for index,row in test.iterrows():
        rtest=[]
        target=row['id']#id作为target
        #全局特征
        gid_num=0
        #source_system_tab
        str_sst=row['source_system_tab']
        if (str_sst in gid_sst):
            rtest.append('%d:%d' % (1,gid_sst.index(str_sst)+1))
            gid_num=gid_num+1
        #source_screen_name
        str_ssn=row['source_screen_name']
        if (str_ssn in gid_ssn):
            rtest.append('%d:%d' % (2,gid_ssn.index(str_ssn)+1))
            gid_num=gid_num+1
        #source_type
        str_st=row['source_type']
        if (str_st in gid_st):
            rtest.append('%d:%d' % (3,gid_st.index(str_st)+1))
            gid_num=gid_num+1
        #用户特征
        uid_num=0
        #msno
        rtest.append('%d:%d' % (1,mid_list.index(row['msno'])+1))
        uid_num=uid_num+1
        #city
        city_value=row['city']
        if city_value and int(city_value)!=0:#不为空
            rtest.append('%d:%d' % (2,int(city_value)))
            uid_num=uid_num+1
        #bd
        bd_value=row['bd']
        if bd_value and 0<int(bd_value)<100:#不为空
            rtest.append('%d:%d' % (3,int(bd_value)))
            uid_num=uid_num+1
        #gender
        gender_value=row['gender']
        if gender_value and int(gender_value)!=0:#不为空
            rtest.append('%d:%d' % (4,int(gender_value)))
            uid_num=uid_num+1
        #registered_via
        registered_via_value=row['registered_via']
        if registered_via_value and int(registered_via_value)!=0:#不为空
            rtest.append('%d:%d' % (5,int(registered_via_value)))
            uid_num=uid_num+1
        #membership_days
        membership_days_value=row['membership_days']
        if membership_days_value and int(membership_days_value)!=0:#不为空
            rtest.append('%d:%d' % (6,int(membership_days_value)))
            uid_num=uid_num+1
        #物品特征
        sid_num=0
        #song_id
        rtest.append('%d:%d' % (1,sid_list.index(row['song_id'])+1))
        sid_num=sid_num+1
        #language
        language_value=row['language']
        if language_value and language_value!='0':#不为空
            rtest.append('%d:%s' % (2,language_value))
            sid_num=sid_num+1
        #song_length
        song_length_value=row['song_length']
        if song_length_value:
            if int(song_length_value)!=0:#不为空
                rtest.append('%d:%d' % (3,int(song_length_value)))
                sid_num=sid_num+1
        #song_year
        song_year_value=row['song_year']
        if song_year_value :
            if song_year_value!='0':#不为空
                rtest.append('%d:%s' % (4,song_year_value))
                sid_num=sid_num+1
        #song_area
        song_area_value=row['song_area']
        if song_area_value:
            rtest.append('%d:%d' % (5,area_list.index(song_area_value)+1))
            sid_num=sid_num+1       
        #genre_ids
        genre_value=row['genre_ids']
        if genre_value:#不为空
            fgenre=str(genre_value).split('|')[0]
            if fgenre!='no' and fgenre!='nan':
                rtest.append('%d:%s' % (6,fgenre))
                sid_num=sid_num+1
        #artist_name
        artist_value=row['artist_name']
        if artist_value!='no':
            fartist=re.split('/|\\\|;|\||&',artist_value)[0]
            if fartist in artist:
                rtest.append('%d:%d' % (7,artist.index(fartist)+1))
                sid_num=sid_num+1          
        #composer
        composer_value=row['composer']
        if composer_value!='no':
            fcomposer=re.split('/|\\\|;|\||&',composer_value)[0]
            if fcomposer in composer:
                rtest.append('%d:%d' % (8,composer.index(fcomposer)+1))
                sid_num=sid_num+1           
        #lyricist
        lyricist_value=row['lyricist']
        if lyricist_value!='no':
            flyricist=re.split('/|\\\|;|\||&',lyricist_value)[0]
            if flyricist in lyricist:
                rtest.append('%d:%d' % (9,lyricist.index(flyricist)+1))
                sid_num=sid_num+1 
        #特征数插入
        rtest.insert(0,sid_num)
        rtest.insert(0,uid_num)
        rtest.insert(0,gid_num)
        rtest.insert(0,target)
        print (rtest)
        uatest.append(rtest)
    uatest.to_csv(homedir + '/ua.test', index=False,encoding='utf8',sep=' ',header=False)
    
    end = time.clock()    
    print('finish all in %s' % str(end - start))       
