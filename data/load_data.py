#coding:utf-8
import pandas as pd
import numpy as np
import talib
# import util.get_3d_data_set as get3d
import datetime
# from scipy.stats import norm
import scipy.signal as signal

from keras.utils.np_utils import to_categorical
# import util.filtered as close_filter
import matplotlib
import os

from data.ProbabilityIndicator import get_probability_indicator

def get_indicators(close_price):
    """ """
    upper, middle, lower = talib.BBANDS(close_price,
                                        timeperiod=26,
                                        # number of non-biased standard deviations from the mean
                                        nbdevup=2,
                                        nbdevdn=2,
                                        # Moving average type: simple moving average here
                                        matype=0)
                                        
    WMA = talib.MA(close_price, 30, matype=2)
    TEMA = talib.MA(close_price, 30, matype=4)
    rsi = talib.RSI(close_price, timeperiod=6)
    macd, macdsignal, macdhist = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
    elas, p = get_probability_indicator(close_price, lmd_1=0.475, lmd_2=0.4)

    mat = close_price
    mat = np.column_stack((mat,upper))
    mat = np.column_stack((mat,middle))
    mat = np.column_stack((mat,lower))
    mat = np.column_stack((mat,WMA))
    mat = np.column_stack((mat,TEMA))
    mat = np.column_stack((mat,rsi))
    mat = np.column_stack((mat,macd))
    mat = np.column_stack((mat,macdsignal))
    mat = np.column_stack((mat,macdhist))
    mat = np.column_stack((mat, p))
    mat = np.column_stack((mat, elas))
    
    return mat
    
def get_legal_input(indicators,statistics_len):  
    """ """
    nan_num = max(np.where(np.isnan(indicators))[0])
    #print 'nan_num',nan_num
    #print 'indicator',indicators.shape
    cut_number = int(max(nan_num,statistics_len))
   # print 'cut_number',cut_number
    legal_input = indicators[cut_number:,:] # the last day has no label, because label comes from the next day.
    
    return legal_input,cut_number
    
#参数close：收盘价，windom_size：滤波窗长度，window_beta:滤波器参数， n:滤波次数
def data_filter_n(close, window_size, window_beta, n):
    
    def data_filter(close, window_size, window_beta):
        window = signal.kaiser(window_size, beta=window_beta)
        close_filtered = signal.convolve(close, window, mode='same') / sum(window)
        close_filtered_modify = list(close_filtered)[int(window_size/2):-int(window_size/2)]#
        return close_filtered_modify
    
    init_datf = [close.values]
    for i in range(n):
        top = list(init_datf[i])[:2]
        end = list(init_datf[i])[len(init_datf[i])-2:]
        middle = list(data_filter(init_datf[i], 5, 2))
        close_filtered_top_end = top + middle + end
        init_datf.append(close_filtered_top_end)
    #输出为n次滤波后的结果
    return  init_datf[-1]
    
#输入参数init_dat是n次滤波后的结果，使用时先调用data_filter_n函数，将data_filter_n函数返回的结果作为init_dat
def tag_data_filter_n(init_dat):
        
    l1 = init_dat[1:]
    l2 = init_dat[:-1]

    mane = [a-b for a, b in zip(l1,l2)]
    y = [i for i in range(len(mane))]
    tag = []
    for i in range(len(mane)):
        if i == 0:
            if mane[i] > 0:
                tag.append(1)
            elif mane[i] < 0:
                tag.append(0)
        elif i > 0:
            if mane[i] > 0:
                tag.append(1)
            elif mane[i] < 0:
                tag.append(0)
            elif mane[i-1] < 0 and mane[i+1] > 0:
                tag.append(1)
            elif mane[i-1] > 0 and mane[i+1] < 0:
                tag.append(0)
    
    return  tag


# 输入参数，data_filter是函数data_filter_n的结果，tag_data_filter是函数tag_data_filter_n的结果，k_threshould是一个涨跌区间内，将当前状态定义为急和缓的阈值
def k_state_tag(data_filter, tag_data_filter, k_threshold):
    change_loc = []
    change_loc.append(0)
    for i in range(1, len(tag_data_filter)):
        if tag_data_filter[i] == tag_data_filter[i - 1]:
            pass
        elif tag_data_filter[i] != tag_data_filter[i - 1]:
            change_loc.append(i)
    if change_loc[-1] == len(tag_data_filter) - 1:
        pass
    else:
        change_loc.append(len(tag_data_filter) - 1)

    change_loc_values = []
    for loc in change_loc:
        change_loc_values.append(data_filter[loc])

    k_s = []
    for i in range(len(change_loc) - 1):
        k = (change_loc_values[i + 1] - change_loc_values[i]) / (change_loc[i + 1] - change_loc[i])
        k_s.append(k)
    t_k = []
    for k in k_s:
        if abs(k) >= k_threshold:
            t_k.append(5)
        else:
            t_k.append(4)

    tag_k = []
    for i in range(len(t_k)):
        a = t_k[i]
        for j in range(change_loc[i], change_loc[i + 1]):
            tag_k.append(a)
    tag_k.append(tag_k[-1])

    return tag_k

def tag_data(close,windom_size,window_beta,n,m,k_threshold):
    init_datf = data_filter_n(close,windom_size,window_beta,n)
    tag_filter = tag_data_filter_n(init_datf)

    init_datf_k = data_filter_n(close, windom_size, window_beta, m)
    tag_filter_k = tag_data_filter_n(init_datf_k)
    tag_k = k_state_tag(init_datf_k,tag_filter_k,k_threshold=k_threshold)

    tupe_list = [(5,1),(5,0),(4,1),(4,0)]
    tag_tuple = zip(tag_k,tag_filter)
    tag = [tupe_list.index(v) for v in tag_tuple]
    return tag,init_datf
    
def get_datasets_2(close,parameter_dict,show_label=True):
    """
    it has already cut the top NTradeDays, which is used for calculating the indicators.
    But the data it returns contain the extra days for tagging data.
    """
    window_size = parameter_dict['window_size']
    window_beta = parameter_dict['window_beta']
    n = parameter_dict['filter_times_of_direciton_object']
    m = parameter_dict['filter_times_of_gradient_object']
    k_threshold = parameter_dict['k_threshold']
    raw_y ,init_datf= tag_data(close,windom_size=window_size,window_beta=window_beta,n=n,m=m,k_threshold=k_threshold)
    raw_y = np.array(raw_y)
    close = np.array(close.tolist())
    pct = np.diff(close)/close[:-1]
    
    raw_x = get_indicators(pct)
   # print 'filtered_close',len(filtered_close),filtered_close[:10]
    
    nan_num = max(np.where(np.isnan(raw_x))[0])+1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    if NTradeDays_for_indicatorCalculation<nan_num+1:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN+1,i.e.%s'%str(nan_num+1))

    x = raw_x[NTradeDays_for_indicatorCalculation-1:]

    labels = raw_y[NTradeDays_for_indicatorCalculation-1:]
    close_for_use = close[NTradeDays_for_indicatorCalculation:]
    filter_data_for_use = init_datf[NTradeDays_for_indicatorCalculation:]
    
    # print 'x',np.where(np.isnan(x))
    # print 'labels',set(labels)
    import matplotlib.pyplot as plt
    labels = labels.astype(int)
    filter_data_for_use = np.array(filter_data_for_use)
    if show_label:
        fig = plt.figure()
        plt.plot(range(close_for_use.shape[0]),filter_data_for_use)
        plt.plot(range(close_for_use.shape[0]),close_for_use)
        plt.scatter(np.where(labels == 0)[0],filter_data_for_use[np.where(labels==0)[0]],marker='o',c='r',label='0',s=30)
        plt.scatter(np.where(labels == 1)[0], filter_data_for_use[np.where(labels == 1)[0]], marker='o',c='y',label='1',s=30)
        plt.scatter(np.where(labels == 2)[0], filter_data_for_use[np.where(labels == 2)[0]], marker='o', c='b', label='2', s=30)
        plt.scatter(np.where(labels == 3)[0], filter_data_for_use[np.where(labels == 3)[0]], marker='o', c='g', label='3', s=30)
        plt.legend()
        plt.show()
    else:
        pass
    # print 'figure plot!'
    return x,labels,close_for_use,filter_data_for_use


def get_balanced_datasets(x,y,parameter_dict):
    nb_class = parameter_dict['nb_class']
    print 'nb_class',nb_class
    
    idx_list = [np.where(y==i)[0] for i in range(nb_class)]
    min_len = min([v.shape[0] for v in idx_list])
    print 'min_len',min_len

    select_id_list = [np.random.choice(range(len(idx_)),min_len).tolist() for idx_ in idx_list]
    balanced_x_list = list(map(lambda idx_,select_id: x[idx_[select_id],:], idx_list,select_id_list))
    balanced_y_list = list(map(lambda idx_,select_id:y[idx_[select_id]], idx_list,select_id_list))
    balanced_x = np.vstack(balanced_x_list)
    # print 'balanced x shape',balanced_x.shape
    train_x = balanced_x.reshape(-1,balanced_x.shape[1],1,1)
    train_y = np.hstack(balanced_y_list)

    return train_x,train_y
    
    
def get_balanced_shuffled_datasets(X,Y,parameter_dict):
    
    train_x,train_y = get_balanced_datasets(X,Y,parameter_dict)

    seed_train = 585
    np.random.seed(seed_train)
    randIdx_array = np.random.permutation(train_x.shape[0])
   # print r
    
    train_x = train_x[randIdx_array,:,:,:]
    train_y = train_y[randIdx_array]
    

    return train_x, train_y
    
    
    
# def get_DataSets(raw_close,statistics_len,ratio_of_sigma):
#
#     X, Y = get_datasets_0(raw_close,statistics_len,ratio_of_sigma)
#
#     train_x,train_y = get_shuffled_datasets(X,Y)
#     print 'train_x,trax_y',train_x.shape,train_y.shape
#
#     return train_x,train_y
    
    
# def get_datasets_0(raw_close,statistics_len,ratio_of_sigma):
#     """ """
#     indicators = get_indicators(raw_close)
#     legal_input,cut_number = get_legal_input(indicators,statistics_len)
#     labels = get_labels(cut_number,statistics_len,raw_close,ratio_of_sigma)
#     X = legal_input
#     #print 'X',X.shape
#     Y = labels
#     #print 'Y',Y.shape
#     time_len = 20
#     X_ = []
#     Y_ = []
#     for i in range(X.shape[0]-time_len+1):
#         sample = X[i:i+time_len,:]
#     #print 'X,Y',X.shape,Y.shape
#         X_.append(sample)
#         Y_.append(Y[time_len+i-1])
#     X_array = np.array(X_)
#     Y_array = np.array(Y_)
#
#     return X_array,Y_array
#
#
# def get_datasets_1(raw_close, statistics_len, ratio_of_sigma):
#     """ """
#     indicators = get_indicators(raw_close)
#     legal_input, cut_number = get_legal_input(indicators, statistics_len)
#     labels = get_labels(cut_number, statistics_len, raw_close, ratio_of_sigma)
#     X = legal_input
#     # print 'X',X.shape
#     Y = labels
#     # print 'Y',Y.shape
#
#     return X, Y
if __name__=='__main__':
   # csv_path = "/home/zqfrgzn05/GenerateData_Model/Htargets/raw_data/original.csv"
    
    csv_path = '/mnt/aidata/生成数据/000905.SH_zscore/20050202-20141231.csv'
    
    raw_pct = pd.read_csv(csv_path,header=None).as_matrix()[:,:-1][0,:]
    #raw_pct = pd.read_csv(csv_path).as_matrix()[:,:-1][:1200,:].reshape(-1)
    print 'raw_pct shape', raw_pct.shape
   
    print  'ratio_of_sigma', '[positive,middle,negative]', 'positive/negaitve','(positive+negative)/all_sample', 'middle/all_sample'
    for v in np.linspace(0.1,2,20):
       
        x,y = get_datasets_0(raw_pct,240,v)
        len_positive = len(np.where(y==1)[0])
        len_negative = len(np.where(y==-1)[0])
        len_middle = len(np.where(y==0)[0])
      
        r_interested = float(len_positive+len_negative)/y.shape[0]
        r_0 = 1 - r_interested
        
        print v,[len_positive,len_middle,len_negative],float(len_positive)/len_negative,r_interested,r_0
        #print 'positive/negaitve:',float(len_positive)/len_negative
        #print '(positive+negative)/all_sample:',r_interested, 'middle/all_sample:',r_0
     
        
   # print 'train_x shape',train_x.shape
   # print 'train_y shape',train_y.shape
   # print 'valid_x shape',valid_x.shape
   # print 'valid_y shape',valid_y.shape
    
    
    
    

    
    
    
    
    
    
    
    

    
        