import merge_data
import mlp_solve
import pandas as pd
from numpy import *
import numpy as np

begin_date = '2019-12-31'
end_date = '2021-07-01'
begin_time = '2019-12-31  0:00:00'
end_time = '2021-07-02  0:00:00'


# %%
mean_D_R = []
min_D_R = []
quantile_D_R = []
std_D_R = []



counter = 0
CHOOSE = [33]
for choose_num in CHOOSE:
    # 数据清洗，生成价格和收益率的表格
    merge_data.mergedata(begin_date, end_date, begin_time, end_time, choose_num)
    # 计算最优组合，boostrap取平均为一次实验
    test_num = 10
    bootsrap_num = 20
    print('counter',counter)

    [mean_Delta_Rate, min_Delta_Rate, quantile_Delta_Rate, Std_Delta_Rate, Q_reault] \
        = mlp_solve.mlp_solve(choose_num, test_num, bootsrap_num, counter)
    counter += 1


    mean_D_R.append(mean_Delta_Rate)
    min_D_R.append(min_Delta_Rate)
    quantile_D_R.append(quantile_Delta_Rate)
    std_D_R.append(Std_Delta_Rate)


    result_delta = pd.DataFrame({'CHOOSE':CHOOSE, 'mean_D_R': mean_D_R, 'min_D_R': min_D_R,
                             'quantile_D_R': quantile_D_R, 'std_D_R': std_D_R})
    result_delta.to_csv('./test_'+str(choose_num)+'/result_delta.csv', mode='w', header=True, index=False)

