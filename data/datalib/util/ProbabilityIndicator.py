#coding:utf-8
import numpy as np
import pandas as pd

def rollstd(x, window_size):
    roll_std_arr = np.zeros(shape = x.shape)
    for i in range(x.shape[0]-window_size+1):
        roll_std_arr[i+window_size-1] = np.std(x[i:i + window_size])
    return roll_std_arr
def rollmean(x, window_size):
    roll_mean_arr = np.zeros(shape = x.shape)
    for i in range(x.shape[0]-window_size+1):
        roll_mean_arr[i + window_size - 1] = np.mean(x[i:i + window_size])
    return roll_mean_arr

def get_probability_indicator(tg_idx,lmd_1,lmd_2):
    p = np.zeros(shape=tg_idx.shape)
    q = rollstd(tg_idx, 20)
    u_1 = 0.725 / 250
    u_2 = -0.125 / 250
    tmp = sma(np.square(q), 25, 'ema')
    q = np.sqrt(tmp)
    p[0] = 0.5
    elas = np.zeros(shape = tg_idx.shape)
    for t in range(tg_idx.shape[0]-1):
        if q[t]!=0:

            g_p = -(lmd_1 + lmd_2) * p[t] + lmd_2 - (u_1 - u_2) * p[t] * (1 - p[t]) * ((u_1 - u_2) * p[t] + u_2 - np.square(q[t]) / 2) / np.square(q[t])
            try:
                tmp = np.log(max(0.1,tg_idx[t+1]+1))
                p[t+1] = min(max(p[t] + g_p + (u_1-u_2) * p[t] * (1 - p[t]) / q[t]**2 * tmp, 0),1)
            except IOError:
                print('calculate error')
            elas[t+1] = 0.01/(u_1-u_2)*p[t]*(1-p[t])*(q[t]**2)
    return elas, p
        # p(t + 1) = min(max(p(t) + g_p + (u_1(1) - u_2(1)) * p(t) * (1 - p(t)) / q(t) ^ 2 * log(tg_idx(t + 1) + 1), 0),
        #                1);

    # for  t=1:tg_idx.shape[0] - 1:
    # g_p = -(lmd_1(1) + lmd_2(1)) * p(t) + lmd_2(1) - (u_1(1) - u_2(1)) * p(t) * (1 - p(t)) * (
    # (u_1(1) - u_2(1)) * p(t) + u_2(1) - q(t) ^ 2 / 2) / q(t) ^ 2;
    # p(t + 1) = min(max(p(t) + g_p + (u_1(1) - u_2(1)) * p(t) * (1 - p(t)) / q(t) ^ 2 * log(tg_idx(t + 1) + 1), 0), 1);
    # end
def lag(x,lag_days):
    y = np.zeros(shape=x.shape)
    y[lag_days:] = x[:-lag_days]
    return y
#    nan_num = np.where(np.isnan(indicators))[0]
def sma(x, m, n):
#   加权平滑方法: x表示数据，m表示加权窗口期，n为‘tw’表示简单时间衰减加权，n为‘ema’表示指数平滑,n为‘sw’表示简单等权平均
#   此处显示详细说明
    if n=='tw':
        y=0
        for i in np.arange(1,m+1):
            y = y + lag(x,i) * (m-i+1)
        y = y/sum(np.arange(1,m+1))

    elif n == 'ema':
        y = np.zeros(shape=x.shape)
        x[np.where(np.isnan(x))] = 0
        x[np.where(np.isinf(x))] = 0
        k = 2.0/(m+1)
        for i in range(1,len(x)):
            y[i] = k*(x[i]-y[i-1])+y[i-1]
    elif n == 'sw':
        y = rollmean(x)

    return y


if __name__ == '__main__':
    # csv_path = "/home/zqfrgzn05/GenerateData_Model/Htargets/raw_data/original.csv"

    #csv_path = '/mnt/aidata/生成数据/000905.SH_zscore/20050202-20141231.csv'

    #raw_close = pd.read_csv(csv,header=None).as_matrix()
    seed = 5
    np.random.seed(seed)
    raw_close = np.random.rand(100)+100
    elas,p = get_probability_indicator(raw_close,lmd_1=0.4,lmd_2=0.475)
    print 'p',p.shape
    dicts = {'raw_close':raw_close.tolist(),'elas':elas.tolist(),'probability':p.tolist()}
    df = pd.DataFrame(dicts)
    df.to_csv('proba_elas.csv')



