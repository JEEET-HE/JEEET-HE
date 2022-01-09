import pandas as pd
import os
from numpy import *
import numpy as np



# %%
def mergedata(begin_date, end_date, begin_time, end_time, choose_num):
    # 数据清洗
    # 以IF和长江保护为基础构建数组
    # The 'Dataset' below contains all the data, both future and etfs.
    # It is constructed by 2 steps, future first.  
    Dataset = pd.read_csv('./Futures_price.csv',                                         #data loading...
                          usecols=['Date', 'IF', 'IC', 'IH'],     
                          dtype={'Date': object, 'IF': np.float64, 'IC': np.float64, 'IH': np.float64},
                          escapechar='/', na_values=r"\N")

    Dataset = Dataset[(Dataset['Date'] >= begin_date) & (Dataset['Date'] <= end_date)]   #data collecting...
    # num of samples
    N = Dataset.shape[0]                                # N represents the length of the futures data
    print(N)



    file_dir = './test_'+ str(choose_num) +'/Tracking_Index'
    files = os.listdir(file_dir)                                             #path = 'test33/Tracking_Index'

    for single_csv in files:
        stock_name = single_csv[:-4]                                        
        path = os.path.join(file_dir + '/', single_csv)
        single_stock = pd.read_csv(path, escapechar='/', na_values=r"", encoding='gbk')
        single_stock = single_stock[(single_stock['Unnamed: 0'] >= begin_time) & (single_stock['Unnamed: 0'] <= end_time)]  
        # got 736lines, data[condition] -> new data
        #print(single_stock,single_stock.shape[0])
        if single_stock.shape[0] == N:
            Dataset[stock_name] = np.array([single_stock[stock_name]]).reshape(-1, 1)
        else:
            print(single_csv, 'not equal')

    Dataset = Dataset.reset_index(drop=True)         #do not know what happened???       
    Dataset.to_csv('./test_'+ str(choose_num) +'/Dataset.csv', mode='w', header=True, index=False, encoding='gbk')
    #print(Dataset)


    # %%
    Price = pd.read_csv('./test_'+ str(choose_num) +'/Dataset.csv', encoding='gbk')
    N = Price.shape[0]

    Date = np.array(Price['Date'])[1:] # Before calculating the pnl_rate, put the first day away.
    Rate_3m = pd.DataFrame({'Date': Date})


    # %%
    # 计算3月累积收益率
    Price.drop(columns=['Date'],inplace=True)

    for column in Price:
        Rate = []
        for i in range(N-1):
            rate = Price.loc[i + 1].at[column] / Price.loc[i].at[column] - 1
            Rate.append(rate)
        Rate_3m[column] = Rate
        #print(Rate_3m)

    Rate_3m.to_csv('./test_'+ str(choose_num) +'/Rate_3m.csv', mode='w', header=True, index=False, encoding='gbk')


## Anyway, Rate_3m is what we want.