import loss_plt   # NOT IMPORTANT
import numpy as np
from gurobipy import *          #integer programming
from qpsolvers import solve_qp  #quadratic programming
from time import perf_counter
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.core.pylabtools import figsize
import matplotlib



# %%
def mlp_solve(choose_num, test_num, bootsrap_num, counter):

    Futures = ['IC','IF']
    # only use 2 futures: IC, IF.
    #print('len',len(Futures))
    # 创建存储中间过程的文件夹
    if not os.path.exists('test_'+str(choose_num)+'/process/'):  # 判断所在目录下是否有该文件名的文件夹
        os.mkdir('test_'+str(choose_num)+'/process/')  # 创建多级目录用mkdirs，单级目录mkdir

    # 读入价格 和 收益率
    Price = pd.read_csv('./test_'+str(choose_num)+'/Dataset.csv', encoding='gbk')
    Rate_3m = pd.read_csv('./test_'+str(choose_num)+'/Rate_3m.csv', encoding='gbk')
    N = Rate_3m.shape[0]
    print(N)
    Price = Price.drop(Price.tail(1).index) #删除最后1行
    # Keep it real!(length of rate should be same with the length of price)



    # 调整价格手数权重，方便取整，期货300，ETF100
    Price[['IF','IH']] = 300 * Price[['IF','IH']]
    Price[['IC']] = 200 * Price[['IC']]
    Price[Price.columns.difference(['Date','IF','IC','IH'])] = 100 * Price[Price.columns.difference(['Date','IF','IC','IH'])]




    # %%
    # 实验
    Mean = pd.DataFrame({'PL_F': [], 'PL_E': [], 'cost_F': [], 
                        'cost_E': [], 'Rate_F': [], 'Rate_E': [], 
                        'Delta_PL': [],'Delta_Rate': []})
    Std = pd.DataFrame(data=None, columns=Mean.columns)
    Min = pd.DataFrame(data=None, columns=Mean.columns)
    Quantile = pd.DataFrame(data=None, columns=Mean.columns)
    # 保存的Q结果与rate表头一致
    # In fact, Q_solution contains the variables we care.
    Q_solution = pd.DataFrame(data=None, columns=Rate_3m.columns)
    Q_solution = Q_solution.drop(columns=['Date','IH'])
    for k in range(test_num):
        print(k)
        # 随机抽取测试集,样本集362，测试集100
        # Actually, here we did not do it RANDOM. 
        Price_test = Price.tail(n=100)
        Price_sample = Price[~Price.index.isin(Price_test.index)]
        Rate_3m_test = Rate_3m.tail(n=100)
        Rate_3m_sample = Rate_3m[~Rate_3m.index.isin(Rate_3m_test.index)]

        # %%
        # 每次实验会重置
        names = np.array(Q_solution.columns)
        QPsolution = pd.DataFrame({'Name': names})
        #print(QPsolution)
        for m in range(bootsrap_num):
            #print(m)
            Num_bootstrap = 250
            # 自抽样250个（1年）
            Price_bst = Price_sample.sample(n=Num_bootstrap, replace=False, random_state=k * m, axis=0)
            Rate_3m_bst = Rate_3m_sample.sample(n=Num_bootstrap, replace=False, random_state=k * m, axis=0)

            #print(Price_bst)
            # 期货的价格和收益率
            Price_F_bst = np.array(Price_bst[Futures])
            Rate_F_bst = np.array(Rate_3m_bst[Futures])
            # 不用上证50
            # Price_F_bst = np.array(Price_bst[['IF', 'IC']])
            # Rate_F_bst = np.array(Rate_3m_bst[['IF', 'IC']])
            PR_F_bst = Price_F_bst * Rate_F_bst # 每日盈亏（一手）

            # ETF的价格和收益率
            Price_bst = Price_bst.drop(columns=['Date','IF','IC','IH'])
            Rate_3m_bst = Rate_3m_bst.drop(columns=['Date','IF','IC','IH'])
            Price_E_bst = np.array(Price_bst)
            Rate_E_bst = np.array(Rate_3m_bst)
            PR_E_bst = Price_E_bst * Rate_E_bst # 每日盈亏（一手）

            # %%
            try:
                # Create a new model
                mlp = Model("mip1")
                # Create variables
                
                #important, we should focus on the variables because the objective is obvious.



                # vtype = GRB.CONTINUOUS / INTEGER / BINARY, name = "IF", lb =, ub =

                # Futures 部分
                ColNames_F = Futures
                #print(ColNames_F)
                createVar_F = locals()  # will自动创建变量名
                PL_F_bst = np.zeros(250)
                for j in range(len(Futures)):
                    print(j)
                    createVar_F['x' + str(j + choose_num + 1)] = mlp.addVar(vtype=GRB.INTEGER, lb=0,
                                                                            name=ColNames_F[j])
                    pl_f = PR_F_bst[:, j] * createVar_F['x' + str(j + choose_num + 1)]# 每手盈亏*手数
                    PL_F_bst = PL_F_bst + pl_f

                #######
                # We can observe:
                # the portfolio(holdings, or we can say variables) will not change during the period(250days).

                # ETF部分
                ColNames_E = Price_bst.columns
                print(ColNames_E)
                createVar_E = locals()  # 自动创建变量名
                PL_E_bst = np.zeros(250)
                Cost_E_bst = np.zeros(1)
                for i in range(choose_num):
                    print(i)
                    print(ColNames_E[i])
                    #if ColNames_E[i] == 'MSCI中国A50互联互通(在岸人民币)' | '300非银' | '300医药' | '科创创业50':
                    if ColNames_E[i] == '300医药':
                        createVar_E['x' + str(i + 1)] = mlp.addVar(vtype=GRB.INTEGER, lb = 3, name=ColNames_E[i])
                    elif ColNames_E[i] == '300非银':
                        createVar_E['x' + str(i + 1)] = mlp.addVar(vtype=GRB.INTEGER, lb=3, name=ColNames_E[i])
                    elif ColNames_E[i] == '科创创业50':
                        createVar_E['x' + str(i + 1)] = mlp.addVar(vtype=GRB.INTEGER, lb=3, name=ColNames_E[i])
                    else:
                        createVar_E['x' + str(i + 1)] = mlp.addVar(vtype=GRB.INTEGER, lb=0, name=ColNames_E[i])
                    #print(createVar_E['x' + str(i+1)])
                    pl_e = Rate_E_bst[:, i] * createVar_E['x' + str(i+1)] * 10000 * 100
                    #print(pl_i)
                    PL_E_bst = PL_E_bst + pl_e
                    #print(PL_E_bst)

                delta = np.square(PL_E_bst - PL_F_bst)

                # Set objective
                mlp.setObjective(np.sum(delta), GRB.MINIMIZE)
                # Add constraint
                # 所有ETF和接近200百万
                Cost = 0
                for i in range(choose_num):
                    Cost = Cost + createVar_E['x' + str(i+1)]
                c1 = mlp.addConstr(Cost >= 200, "c0")

                mlp.optimize()
                Q_solve = []
                for v in mlp.getVars():
                    print(v.varName, v.x)
                    Q_solve.append(v.x)
                print('Obj:', mlp.objVal)

                # 存储结果
                QPsolution['Q' + str(m)] = Q_solve

            except GurobiError:
                print('Error reported')


        #print(QPsolution)
        QPsolution.to_csv('./test_'+str(choose_num)+'/process/Q_Test'+str(k)+'.csv', mode='w', header=True, index=False, encoding='gbk')

        # %%
        # 对解求平均数
        QPsolution = QPsolution.drop(columns=['Name'])
        QP = np.array(QPsolution)
        Q = QP.mean(axis=1)
        Q = np.around(Q)
        #print(Q,Q.shape)


        # %%
        # 在测试集上看loss效果
        Date = np.array(Rate_3m_test['Date'])
        Price_F_test = np.array(Price_test[Futures])
        Rate_F_test = np.array(Rate_3m_test[Futures])
        PR_F_test = Price_F_test * Rate_F_test

        # IF+股票的价格和收益率
        Price_test = Price_test.drop(columns=['Date', 'IF', 'IC', 'IH'])
        Rate_3m_test = Rate_3m_test.drop(columns=['Date', 'IF', 'IC', 'IH'])
        Price_E_test = np.array(Price_test)
        Rate_E_test = np.array(Rate_3m_test)

        # 计算P/L
        PL_F = np.dot(PR_F_test, Q[0:len(Futures)]) #前3行
        PL_E = np.dot(Rate_E_test, Q[len(Futures):]) * 10000 * 100  #从第4个Q开始

        PL = pd.DataFrame({'Date': Date, 'PL_F': PL_F, 'PL_E': PL_E})

        # 计算cost
        cost_F = np.dot(Price_F_test, Q[0:len(Futures)])
        print(np.sum(Q[len(Futures):]))
        cost_E = np.sum(Q[len(Futures):]) * 10000 * 100

        # 计算rate
        Rate_F = PL_F/cost_F
        Rate_E = PL_E/cost_E


        PL['cost_F'], PL['cost_E'], PL['Rate_F'], PL['Rate_E'] = [cost_F, cost_E, Rate_F, Rate_E]

        # 计算对冲误差
        Delta_PL = PL_E - PL_F
        Delta_Rate = Rate_E - Rate_F

        PL['Delta_PL'], PL['Delta_Rate'] = [Delta_PL, Delta_Rate]

        PL.to_csv('./test_'+str(choose_num)+'/process/PL_Test'+str(k)+'.csv', mode='w', header=True, index=False, encoding='gbk')

        # %%
        # 作图
        loss_plt.PL_plt(Date, PL_F, PL_E, Delta_PL, k, choose_num)
        loss_plt.Rate_plt(Date, Rate_F, Rate_E, Delta_Rate, k, choose_num)


        # 输出统计分析
        PLT = PL.drop(columns=['Date'])
        PLT = np.array(PLT)
        PL_std = PLT.std(axis=0)  # 标准差
        PL_mean = PLT.mean(axis=0)  # 均值
        PL_min = PLT.min(axis=0) # 最小值
        PL_q = np.percentile(PLT, 5, axis=0, interpolation='midpoint') # 5%分位数
        #print(PL_mean.shape)

        Q_solution.loc[k], Mean.loc[k], Std.loc[k], Min.loc[k], Quantile.loc[k]= [Q, PL_mean,PL_std,PL_min,PL_q]


    Mean.to_csv('./test_'+str(choose_num)+'/Mean.csv', mode='w', header=True, index=False)
    Std.to_csv('./test_'+str(choose_num)+'/Std.csv', mode='w', header=True, index=False)
    Min.to_csv('./test_'+str(choose_num)+'/Min.csv', mode='w', header=True, index=False)
    Quantile.to_csv('./test_'+str(choose_num)+'/Quantile.csv', mode='w', header=True, index=False)
    Q_solution.to_csv('./test_'+str(choose_num)+'/Q_solution.csv', mode='w', header=True, index=False, encoding='gbk')

    # 计算平均数作为本组实验结果
    mean_Delta_Rate = Mean['Delta_Rate'].mean()
    min_Delta_Rate = Min['Delta_Rate'].mean()
    quantile_Delta_Rate = Quantile['Delta_Rate'].mean()
    Std_Delta_Rate = Std['Delta_Rate'].mean()
    Q_reault = Q_solution.mean()

    result_Q = pd.DataFrame(data=None, columns=Q_solution.columns)
    result_Q = result_Q.append(Q_reault, ignore_index=True)

    if counter == 0:
        result_Q.to_csv('./test_'+str(choose_num)+'/result_Q.csv', mode='w', header=True, index=False, encoding='gbk')
    else:
        result_Q.to_csv('./test_'+str(choose_num)+'/result_Q.csv', mode='a', header=False, index=False, encoding='gbk')


    return(mean_Delta_Rate, min_Delta_Rate, quantile_Delta_Rate, Std_Delta_Rate, Q_reault)

