#coding:utf-8
import pandas as pd
import numpy as np
import talib
import util.get_3d_data_set as get3d
import datetime
# from scipy.stats import norm


from keras.utils.np_utils import to_categorical
import util.filtered as close_filter
import matplotlib
import os
from data.datalib.util.ProbabilityIndicator import get_probability_indicator


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
    
def get_filtered_data(close):
    index_close = close
    close_df = pd.DataFrame(index_close, columns=['close'])
     # = MyFilter(data_use, window_width=10, draw_graph=False)
    filtered_data = close_filter.MyFilter(close_df, window_width=10, draw_graph=False)
    # df = pd.DataFrame(filtered_data)
    #df.to_csv(self.filtered_close_path)
    return filtered_data

def get_datasets_2(close,pct, parameter_dict):
    """ """
    filtered_close = get_filtered_data(close).tolist()
    indicator_cutDays = parameter_dict['indicator_cutDays']
   # print 'filtered_close',len(filtered_close),filtered_close[:10]
    t = range(len(filtered_close))
    # print 't',t
    knee_t, knee_close = get3d.knee_point(t, filtered_close)
    print 'knee_t',knee_t
    break_t, break_p = get3d.break_point(0.01, knee_t, knee_close)
    break_t = np.array(break_t)

    break_t = break_t[np.where(break_t > indicator_cutDays-5)]
    #print 'break_t',break_t
    data_idx = break_t + 5
    print 'data_idx[0]', data_idx[-1]
    labels_range = data_idx-data_idx[0]
    print 'break_idx',data_idx[-1]

    label_idx = zip(labels_range[:-1], labels_range[1:])


    raw_3d_data = get_indicators(pct)
    #print 'raw_3d_data',raw_3d_data.shape
   # print data_idx
    data_for_use = raw_3d_data[data_idx[0]:data_idx[-1]]

    filtered_close_for_use = filtered_close[break_t[0]:break_t[-1]]

    close_for_use = close[5:-4][break_t[0]:break_t[-1]]

    #time_serias = self.raw_kdata[5:-4, 0][break_t[0]:break_t[-1]]

    labels = np.zeros(data_for_use.shape[0])

    if filtered_close_for_use[1] - filtered_close_for_use[0] > 0:

        for i, idx in enumerate(label_idx):

            if i % 2 == 0:
                labels[idx[0]:idx[0] + 3] = 1
                labels[idx[0] + 3:idx[1] - 3] = 2
                labels[idx[1] - 3:idx[1]] = 3
            else:
                labels[idx[0]:idx[0] + 3] = 3
                labels[idx[0] + 3:idx[1] - 3] = 4
                labels[idx[1] - 3:idx[1]] = 1

    else:

        for i, idx in enumerate(label_idx):

            if i % 2 == 0:
                labels[idx[0]:idx[0] + 3] = 3
                labels[idx[0] + 3:idx[1] - 3] = 4
                labels[idx[1] - 3:idx[1]] = 1
            else:
                labels[idx[0]:idx[0] + 3] = 1
                labels[idx[0] + 3:idx[1] - 3] = 2
                labels[idx[1] - 3:idx[1]] = 3
    print 'data_for_use','label',data_for_use.shape,labels.shape
    print 'filtered_data,close_for_use',(len(filtered_close_for_use),close_for_use.shape[0])
    # print 'labels',set(labels)
    labels = labels.astype(int)
    import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(range(len(close_for_use)),close_for_use,c='r')
    # plt.plot(range(len(close_for_use)), filtered_close_for_use, c='b')
    # # print np.where(labels==1)[0]
    # # print np.where(labels == 2)[0]
    # # print np.where(labels == 3)[0]
    # # print np.where(labels == 4)[0]
    # # print type(filtered_close_for_use)
    # import matplotlib.pyplot as plt
    # filtered_close_for_use = np.array(filtered_close_for_use)
    labels = labels -1
    # plt.scatter(np.where(labels == 0)[0], filtered_close_for_use[np.where(labels == 0)[0]], marker='o', c='g', label='0')
    # plt.scatter(np.where(labels == 1)[0],filtered_close_for_use[np.where(labels==1)[0]],marker='o',c='r',label = '1')
    # plt.scatter(np.where(labels == 2)[0], filtered_close_for_use[np.where(labels == 2)[0]], marker='o',c='y',label = '2')
    # plt.scatter(np.where(labels == 3)[0], filtered_close_for_use[np.where(labels == 3)[0]], marker='o',c='b',label = '3')
    # plt.legend()

    X = data_for_use
    # print 'X',X.shape
    Y = labels
    time_len = 20
    X_ = []
    Y_ = []
    for i in range(X.shape[0] - time_len + 1):
        sample = X[i:i + time_len, :]
        # print 'X,Y',X.shape,Y.shape
        X_.append(sample)
        Y_.append(Y[time_len + i - 1])
    X_array = np.array(X_)
    Y_array = np.array(Y_)

    plt.show()
    ## test data consistent
    X_array = X_array.reshape(-1, X_array.shape[1], X_array.shape[2], 1).transpose(0, 2, 1, 3)
    return X_array, Y_array, filtered_close_for_use,close_for_use
    

def get_datasets_0(raw_close,statistics_len,ratio_of_sigma):
    """ """
    indicators = get_indicators(raw_close)
    legal_input,cut_number = get_legal_input(indicators,statistics_len)
    labels = get_labels(cut_number,statistics_len,raw_close,ratio_of_sigma)
    X = legal_input
    #print 'X',X.shape
    Y = labels 
    #print 'Y',Y.shape
    time_len = 20
    X_ = []
    Y_ = []
    for i in range(X.shape[0]-time_len+1):
        sample = X[i:i+time_len,:]
    #print 'X,Y',X.shape,Y.shape
        X_.append(sample)
        Y_.append(Y[time_len+i-1])
    X_array = np.array(X_)
    Y_array = np.array(Y_)
    
    return X_array,Y_array


def get_datasets_1(raw_close, statistics_len, ratio_of_sigma):
    """ """
    indicators = get_indicators(raw_close)
    legal_input, cut_number = get_legal_input(indicators, statistics_len)
    labels = get_labels(cut_number, statistics_len, raw_close, ratio_of_sigma)
    X = legal_input
    # print 'X',X.shape
    Y = labels
    # print 'Y',Y.shape

    return X, Y
    
def get_balanced_datasets(x,y):
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y== 1)[0]
    idx_2 = np.where(y== 2)[0]
    idx_3 = np.where(y == 3)[0]

    #print len(idx_0),len(idx_1),len(idx_2),len(idx_3),len(idx_4)
    min_len = min([idx_0.shape[0],idx_1.shape[0],idx_2.shape[0],idx_3.shape[0]])
    print 'min_len',min_len
    #print 'min_len',min_len

    select_idx0 = np.random.choice(range(len(idx_0)), min_len).tolist()
    select_idx1 = np.random.choice(range(len(idx_1)),min_len).tolist()
    select_idx2 = np.random.choice(range(len(idx_2)),min_len).tolist()
    select_idx3 = np.random.choice(range(len(idx_3)), min_len).tolist()

    balanced_x_0 = x[idx_0[select_idx0], :, :]
    balanced_x_1 = x[idx_1[select_idx1],:,:]
    balanced_x_2 = x[idx_2[select_idx2],:,:]
    balanced_x_3 = x[idx_3[select_idx3],:,:]

    
    balanced_y_0 = y[idx_0[select_idx0]]
    balanced_y_1 = y[idx_1[select_idx1]]
    balanced_y_2 = y[idx_2[select_idx2]]
    balanced_y_3 = y[idx_3[select_idx3]]

    train_balanced_x = np.vstack((balanced_x_0,balanced_x_1,balanced_x_2,balanced_x_3))
   # print 'balanced x shape',balanced_x.shape  
    #train_x = train_balanced_x.reshape(-1,train_balanced_x.shape[1],train_balanced_x.shape[2],1).transpose(0,2,1,3)
    train_x = train_balanced_x
    train_y = np.hstack((balanced_y_0,balanced_y_1,balanced_y_2,balanced_y_3))
    
    return train_x,train_y
    
    
def get_balanced_shuffled_datasets(X,Y):
    
    train_x,train_y = get_balanced_datasets(X,Y)

    seed_train = 585
    np.random.seed(seed_train)
    r = np.random.permutation(train_x.shape[0])
   # print r
    
    train_x = train_x[r,:,:,:]
    train_y = train_y[r]
    

    return train_x, train_y
    
    
    
def get_DataSets(raw_close,statistics_len,ratio_of_sigma):
    
    X, Y = get_datasets_0(raw_close,statistics_len,ratio_of_sigma)
    
    train_x,train_y = get_shuffled_datasets(X,Y)
    print 'train_x,trax_y',train_x.shape,train_y.shape

    return train_x,train_y 
    
    
    
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
    
    
    
    

    
    
    
    
    
    
    
    

    
        