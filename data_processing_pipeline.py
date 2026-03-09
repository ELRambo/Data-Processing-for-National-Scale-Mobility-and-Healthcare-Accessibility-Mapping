# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:07:29 2022

@author: 10449
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:54:58 2022

@author: 10449
"""

# In[1]

import os

import numpy as np
import pandas as pd

from numba import jit
from numba.typed import List

import datetime as dt
from datetime import datetime

import tkinter
from tkinter import messagebox
import winsound

os.chdir('F:\\Data')

# In[数据更新]

def renew(df):
    # convert to typed_list for numba
    # convert numbers

    lnt_col = df['经度']
    lnt_col = lnt_col.to_list()
    typed_lnt = List()
    for x in lnt_col:
        typed_lnt.append(x)

    lat_col = df['纬度']
    lat_col = lat_col.to_list()
    typed_lat = List()
    for x in lat_col:
        typed_lat.append(x)
            
    size_col = result['size']
    size_col = size_col.to_list()
    typed_size = List()
    for x in size_col:
        typed_size.append(x)

    stop_time = df['场景停留时长']
    stop_time = stop_time.to_list()
    typed_stop = List()
    for x in stop_time:
        typed_stop.append(x)


    # convert datetimes
     
    start_time = df['开始时间']
    start_time = start_time.to_list()
    typed_start = List()
    for x in start_time:
        typed_start.append(np.datetime64(x))

    end_time = df['结束时间']
    end_time = end_time.to_list()
    typed_end = List()
    for x in end_time:
        typed_end.append(np.datetime64(x))


    # convert strings

    uuid = df['脱敏ID']
    uuid = uuid.to_list()
    typed_id = List()
    for x in uuid:
        typed_id.append(x)
        

    length = len(result)
    length2 = len(df)
    
    return typed_lnt, typed_lat, typed_start, typed_end, typed_stop, typed_size, uuid, length, length2

# In[home_loc]

@jit(nopython=True)

def home_loc(lnt_col, lat_col, size, length):

    # typed_list for numba
    home_lnt_li = List()
    home_lat_li = List()

    for i in np.arange(0, length):

        # print('home:', i)

        row = size[i]-1

        lnt = lnt_col[row]
        lat = lat_col[row]

        home_lnt_li.append(lnt)
        home_lat_li.append(lat)

        del lnt_col[0: row+1]
        del lat_col[0: row+1]

    return home_lnt_li, home_lat_li

# In[work_loc]

@jit(nopython=True)

def work_loc(lnt_col, lat_col, morning, evening, start_time, end_time, stop_time, size, length):
    
    # typed_list for numba
    work_lnt_li = []
    work_lat_li = []
    
    for j in np.arange(0, length):
        
        # print('work:',j)
        
        stop_max = stop_time[0]  # 记录最大停留时间
        lnt = 0
        lat = 0
        
        row = size[j]-1
        
        for i in np.arange(0, row):
            
            if(stop_time[i] >= stop_max 
                and start_time[i] < evening      # 判断开始时间是否在白天工作时间
                and end_time[i] > morning):
                
                stop_max = stop_time[i]
                lnt = lnt_col[i]
                lat = lat_col[i]
            
        work_lnt_li.append(lnt)
        work_lat_li.append(lat)
        
        del lnt_col[0: row+1]
        del lat_col[0: row+1]
    
    return work_lnt_li, work_lat_li

# In[commute_extract]

@jit(nopython=True)

def commute_extract(uuid, q1,q2,q3,q4, start_time, end_time, length2):
    
    commute_start = []
    commute_end = []
    id_li = []
    
    for i in np.arange(0,length2):
        
        # print('extract', i)
        
        if (q1<=start_time[i]<=q2 
            or q3<=start_time[i]<=q4):
            
            commute_start.append(start_time[i])
            commute_end.append(end_time[i])
            id_li.append(uuid[i])
    
    #print(commute_time)
    return id_li, commute_start, commute_end
    
# In[commute_time]

@jit(nopython=True)

def commute_time(time_li, size_li, time0, time1m):
    
    commute_time_li = []

    for j in np.arange(0,len(size_li)):
        
        # print('time',j)
        
        row = size_li[j]-1
        
        delta = time0 / time1m
        
        if (row != 1):
            delta = (time_li[row] - time_li[0]) / time1m
        else:
            delta
            
        commute_time_li.append(delta)
        
        del time_li[0: row+1]
        
    if len(commute_time_li):
        return commute_time_li

# In[]

start = dt.datetime.now()

# In[read data]

for i in range(0,20):
    
    print('Processing data: ', i)
    
     
    print('Reading csv...')
    if(i<10):
        count = '0' + str(i)
    else:
        count = str(i)
    Input = '2022-05\\20220530\\2022-05-30_' + count + '.csv'
    Output = 'Data_results\\2022-05\\20220530\\'
    data = pd.read_csv(Input, low_memory=False, encoding='utf-8', on_bad_lines=('skip'))
    
    
    print('Filtering & converting')
    df = data.iloc[:, :6]
    df.columns = ['脱敏ID', '经度', '纬度', '开始时间', '结束时间','场景停留时长']
    # df['城市'] = data.iloc[:, 10]

    # convert dtype: object to numeric & datetime

    if (type(df['经度'].head())!='float64'):
        df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    if (type(df['纬度'].head())!='float64'):
        df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    if (type(df['场景停留时长'].head())!='float64'):
        df['场景停留时长'] = pd.to_numeric(df['场景停留时长'], errors='coerce')

    if(type(df['开始时间'].head())!='datetime64'):
        df['开始时间'] = pd.to_datetime(df['开始时间'], errors='coerce')
    if(type(df['结束时间'].head())!='datetime64'):
        df['结束时间'] = pd.to_datetime(df['结束时间'], errors='coerce')

    df = df.dropna(subset=['结束时间'])

    df = df.loc[(113.751647 < df['经度']) & (df['经度'] < 114.622924) & 
                (22.400047 < df['纬度']) & (df['纬度'] < 22.855425)]
    
    # df = df.loc[(22.400047 < df['纬度']) & (df['纬度'] < 22.855425)]
    # df = df[df['城市']=='深圳市']
    
    df = df.loc[df['场景停留时长'] > 0]
    
    
    
    # Create timestamps
    
    date = df['开始时间'].iloc[0].date()
    
    morning = np.datetime64(str(date)+'T07:00') 
    evening = np.datetime64(str(date)+'T18:00') 

    q1 = np.datetime64(str(date)+'T06:00') 
    q2 = np.datetime64(str(date)+'T10:00') 
    q3 = np.datetime64(str(date)+'T19:00') 
    q4 = np.datetime64(str(date)+'T21:00') 
    
    t0m = np.timedelta64(0, 'm')
    t1m = np.timedelta64(1, 'm')
    
    
    # Create indices
    
    df_size = df.pivot_table(index=['脱敏ID', '开始时间', '结束时间'])
    # print(df_size)
    size = df.groupby(['脱敏ID', pd.Grouper(key='开始时间', axis=0,
                                                freq='1D', sort=True)]).size()  # series
    result = size.to_frame()
    result.columns = ['size']
    
    
    
    print('Calculating home loc...')
    
    lnt_col, lat_col, start_time, end_time, stop_time, size, uuid, length, length2 = renew(df)

    hlnt, hlat = home_loc(lnt_col, lat_col, size, length)

    result['home_lnt'] = hlnt
    result['home_lat'] = hlat
    
    
    
    print('Calculating work loc...')
    
    lnt_col, lat_col, start_time, end_time, stop_time, size, uuid, length, length2 = renew(df)

    wlnt, wlat = work_loc(lnt_col, lat_col, morning, evening, 
                          start_time, end_time, stop_time, size, length)

    result['work_lnt'] = wlnt
    result['work_lat'] = wlat
    result = result.loc[result['work_lat']!=0]
    
    if(i==0):
        result = result.iloc[:-1, :]
    else:
        result = result.iloc[1:-1, :]
    
    result.to_csv(Output + str(i) + '_loc.csv')
    
    
    
    print('Extracting commute points...')
    
    lnt_col, lat_col, start_time, end_time, stop_time, size, uuid, length, length2 = renew(df)

    id_li, start_li, end_li = commute_extract(uuid, q1,q2,q3,q4, start_time, end_time, length2)
    
    id_li_copy = id_li.copy()
    start_li_copy = start_li.copy()
    end_li_copy = end_li.copy()

    data = {'脱敏ID': id_li_copy,
            '开始时间': start_li_copy,
            '结束时间': end_li_copy }
    commute_df = pd.DataFrame(data)

    commute_df_index = commute_df.pivot_table(index=['脱敏ID','开始时间','结束时间'])
    # print(len(commute_df_index))

    size = commute_df_index.groupby(['脱敏ID', pd.Grouper(level='开始时间', axis=0, 
                          freq='12H', sort=True)]).size()

    result2 = size.to_frame()
    result2.columns = ['size']
    size_li = result2['size'].to_list()
    
    
    
    print('Calculating commute time...')
    
    id_li_copy = id_li.copy()
    start_li_copy = start_li.copy()
    end_li_copy = end_li.copy()

    data = {'脱敏ID': id_li_copy,
            '开始时间': start_li_copy,
            '结束时间': end_li_copy }
    commute_df = pd.DataFrame(data)

    commute_df_index = commute_df.pivot_table(index=['脱敏ID','开始时间','结束时间'])
    # print(len(commute_df_index))

    size = commute_df_index.groupby(['脱敏ID', pd.Grouper(level='开始时间', axis=0, 
                          freq='12H', sort=True)]).size()

    result2 = size.to_frame()
    result2.columns = ['size']
    size_li = result2['size'].to_list()
    
    time = commute_df_index.reset_index()['开始时间'].to_list()
    if(len(time)):
        
        time_li = List()
        
        for x in time:
            time_li.append(np.datetime64(x))

    commute_time_li = commute_time(time_li, size_li, t0m, t1m)

    result2['commute_time/min'] = commute_time_li
    result2 = result2.drop(result2[result2['commute_time/min']==0].index)
    
    if(i==0):
        result = result.iloc[:-1, :]
    else:
        result = result.iloc[1:-1, :]
    
    result2.to_csv(Output + str(i) + '_commute.csv')
    
    print('')

# In[End]

end = dt.datetime.now()

print('executed:', end - start)

root = tkinter.Tk()
root.withdraw()
messagebox.showinfo('prompt','completed')
winsound.MessageBeep()