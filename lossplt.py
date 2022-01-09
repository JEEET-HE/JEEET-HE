import pandas as pd
from numpy import *
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.core.pylabtools import figsize
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体['KaiTi'] ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
matplotlib.rcParams['savefig.dpi'] = 100  # 图片像素
matplotlib.rcParams['figure.dpi'] = 100  # 分辨率


def PL_plt(Date, PL_F, PL_E, Delta_PL, k, choose_num):
    # 每20天显示一次坐标刻度
    Time = []
    for i in range(Date.shape[0]):
        if i % 20 == 0:
            time = Date[i]
        else:
            time = ''
        Time.append(time)

    figsize(36, 18)  # 设置figsize
    gs = gridspec.GridSpec(2, 3)

    ax11 = plt.subplot(gs[0, :3])
    ax11.plot(Date, PL_F, color="green",label="Futures", alpha=0.6)
    ax11.plot(Date, PL_E, color="dodgerblue",label="ETFs", alpha=0.6)
    ax11.plot(Date, Delta_PL, color="red", label="Delta", alpha=0.6)

    # 设置坐标轴
    my_x_ticks = np.arange(0,Date.shape[0])  # 每5天一条网格线
    x_txt = Time
    ax11.set_xticks(my_x_ticks)
    ax11.set_xticklabels(x_txt, fontsize=30)

    ax11.spines['right'].set_color('none')
    ax11.spines['top'].set_color('none')

    for size in ax11.get_yticklabels():  # 获取y轴上所有坐标，并设置字号
        size.set_fontsize('40')

    ax11.set_xlabel('date', fontsize=30)
    ax11.set_ylabel('P/L(￥)', fontsize=30)
    ax11.legend(loc=1,borderaxespad = 0.1,fontsize=30)

    ax12 = plt.subplot(gs[1, :3])
    ax12.plot(Date, Delta_PL, color="red", label="Delta", alpha=0.6)

    # 设置坐标轴
    ax12.set_xticks(my_x_ticks)
    ax12.set_xticklabels(x_txt, fontsize=30)

    ax12.spines['right'].set_color('none')
    ax12.spines['top'].set_color('none')

    for size in ax12.get_yticklabels():  # 获取y轴上所有坐标，并设置字号
        size.set_fontsize('40')

    ax12.set_xlabel('date', fontsize=30)
    ax12.set_ylabel('Delta(￥)', fontsize=30)
    ax12.legend(loc=1, borderaxespad=0.1, fontsize=30)
    # 添加注释
    text = '''
                    注：
                    上图为股指期货和ETF组合各自的1日损益值，
                    下图为对冲误差绝对损益值:PnL(ETFs)-PnL(Futures)
                 '''
    plt.figtext(0, 0, text,
                fontdict=dict(family='KaiTi', weight='light', style='oblique', fontsize=30, color='dimgrey'))

    name = './test_'+str(choose_num)+'/process/PL_Test' + str(k)
    plt.title(name,fontsize = 20)
    plt.savefig('./test_'+str(choose_num)+'/process/PL' + str(k) +'.png')
    plt.show()

def Rate_plt(Date, Rate_F, Rate_E, Delta_Rate, k, choose_num):
    # 每20天显示一次坐标刻度
    Time = []
    for i in range(Date.shape[0]):
        if i % 20 == 0:
            time = Date[i]
        else:
            time = ''
        Time.append(time)

    figsize(36, 18)  # 设置figsize
    gs = gridspec.GridSpec(2, 3)

    ax11 = plt.subplot(gs[0, :3])
    ax11.plot(Date, Rate_F, color="green",label="Futures", alpha=0.6)
    ax11.plot(Date, Rate_E, color="dodgerblue",label="ETFs", alpha=0.6)
    ax11.plot(Date, Delta_Rate, color="red", label="Delta", alpha=0.6)

    # 设置坐标轴
    my_x_ticks = np.arange(0,Date.shape[0])  # 每5天一条网格线
    x_txt = Time
    ax11.set_xticks(my_x_ticks)
    ax11.set_xticklabels(x_txt, fontsize=30)

    ax11.spines['right'].set_color('none')
    ax11.spines['top'].set_color('none')

    for size in ax11.get_yticklabels():  # 获取y轴上所有坐标，并设置字号
        size.set_fontsize('40')

    ax11.set_xlabel('date', fontsize=30)
    ax11.set_ylabel('Rate(%)', fontsize=30)
    ax11.legend(loc=1,borderaxespad = 0.1,fontsize=30)

    ax12 = plt.subplot(gs[1, :3])
    ax12.plot(Date, Delta_Rate, color="red", label="Delta", alpha=0.6)

    # 设置坐标轴
    ax12.set_xticks(my_x_ticks)
    ax12.set_xticklabels(x_txt, fontsize=30)

    ax12.spines['right'].set_color('none')
    ax12.spines['top'].set_color('none')

    for size in ax12.get_yticklabels():  # 获取y轴上所有坐标，并设置字号
        size.set_fontsize('40')

    ax12.set_xlabel('date', fontsize=30)
    ax12.set_ylabel('Delta(%)', fontsize=30)
    ax12.legend(loc=1, borderaxespad=0.1, fontsize=30)
    # 添加注释
    text = '''
                    注：
                    上图为股指期货和ETF组合各自的损益率，
                    下图为对冲误差绝对损益值:Rate(ETFs)-Rate(Futures)
                 '''
    plt.figtext(0, 0, text,
                fontdict=dict(family='KaiTi', weight='light', style='oblique', fontsize=30, color='dimgrey'))

    name = './test_'+str(choose_num)+'/process/Rate_Test' + str(k)
    plt.title(name,fontsize = 30)
    plt.savefig('./test_'+str(choose_num)+'/process/Rate' + str(k) +'.png')
    plt.show()