#coding:utf-8
import pandas as pd
import numpy as np
import talib
# import util.get_3d_data_set as get3d
import datetime
# from scipy.stats import norm
import scipy.signal as signal
import datalib.util.get_3d_data_set as get3d
from keras.utils.np_utils import to_categorical
# import util.filtered as close_filter
import matplotlib
import os

from data.ProbabilityIndicator import get_probability_indicator, rollstd

slope_type = {'sharp':5,'gentle':4}
trend_type = {'up': 1, 'down': 0}
def get_indicators_close(close_price):
    """ """
    upper, middle, lower = talib.BBANDS(close_price,
                                        timeperiod=13,
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

    # close_price = np.diff(close_price) / close_price[:-1]
    # upper = np.diff(upper) / upper[:-1]
    # middle = np.diff(middle) / middle[:-1]
    # WMA = np.diff(WMA) / WMA[:-1]
    # TEMA = np.diff(TEMA) / TEMA[:-1]
    # rsi = np.diff(rsi) / rsi[:-1]
    # macd = np.diff(macd) / macd[:-1]
    # macdsignal = np.diff(macdsignal) / macdsignal[:-1]
    # macdhist = np.diff(macdhist) / macdhist[:-1]
    # p = np.diff(p) / p[:-1]
    # elas = np.diff(elas) / elas[:-1]

    mat = close_price
    # mat = np.column_stack((mat,upper))
    # #mat = upper
    # mat = np.column_stack((mat,middle))
    # mat = np.column_stack((mat,lower))

    mat = np.column_stack((mat,WMA))
    mat = np.column_stack((mat,TEMA))
    mat = np.column_stack((mat,rsi))
    #mat = macd
    mat = np.column_stack((mat,macd))
    mat = np.column_stack((mat,macdsignal))
    mat = np.column_stack((mat,macdhist))
    mat = np.column_stack((mat, p))
    mat = np.column_stack((mat, elas))
    
    return mat


def get_indicators_return(close_return):
    """ """
    upper, middle, lower = talib.BBANDS(close_return,
                                        timeperiod=26,
                                        # number of non-biased standard deviations from the mean
                                        nbdevup=2,
                                        nbdevdn=2,
                                        # Moving average type: simple moving average here
                                        matype=0)
    upper1, middle1, lower1 = talib.BBANDS(close_return,
                                           timeperiod=13,
                                           # number of non-biased standard deviations from the mean
                                           nbdevup=3,
                                           nbdevdn=3,
                                           # Moving average type: simple moving average here
                                           matype=0)
    upper2, middle2, lower2 = talib.BBANDS(close_return,
                                           timeperiod=5,
                                           # number of non-biased standard deviations from the mean
                                           nbdevup=3,
                                           nbdevdn=3,
                                           # Moving average type: simple moving average here
                                           matype=0)


    WMA = talib.MA(close_return, 30, matype=2)
    TEMA = talib.MA(close_return, 30, matype=4)

    rsi = talib.RSI(close_return, timeperiod=6)

    macd, macdsignal, macdhist = talib.MACD(close_return, fastperiod=12, slowperiod=26, signalperiod=9)
    elas, p = get_probability_indicator(close_return, lmd_1=0.475, lmd_2=0.4)

    mat = close_return
    mat = np.column_stack((mat, upper))
    mat = np.column_stack((mat, middle))
    mat = np.column_stack((mat, lower))

    mat = np.column_stack((mat, upper1))
    mat = np.column_stack((mat, middle1))
    mat = np.column_stack((mat, lower1))

    mat = np.column_stack((mat, upper2))
    mat = np.column_stack((mat, middle2))
    mat = np.column_stack((mat, lower2))

    mat = np.column_stack((mat, WMA))
    mat = np.column_stack((mat, TEMA))
    mat = np.column_stack((mat, rsi))
    mat = np.column_stack((mat, macd))
    mat = np.column_stack((mat, macdsignal))
    mat = np.column_stack((mat, macdhist))
    mat = np.column_stack((mat, p))
    mat = np.column_stack((mat, elas))

    return mat

def get_indicators(close_price,close_return):
    #mat1 = get_indicators_close(close_price)
    mat2 = get_indicators_return(close_return)
    #mat = np.column_stack((mat1, mat2))
    mat = mat2
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


def data_filter_nTimes(close, window_size, window_beta, filterTimes):
    def data_oneTime_filter(close, window_size, window_beta):

        window_array = signal.kaiser(window_size, beta=window_beta)
        close_filtered = signal.convolve(close, window_array, mode='same') / sum(window_array)
        if window_size % 2 == 0:
            cutExtraDays_for_FilteredClose = list(close_filtered)[int(window_size / 2):-int(window_size / 2 - 1)]  #
        else:
            cutExtraDays_for_FilteredClose = list(close_filtered)[int(window_size / 2):-int(window_size / 2)]  #
        return cutExtraDays_for_FilteredClose

    extraTradeDays_beforeStartTime = int(window_size/2)
    if window_size%2==0:
        extraTradeDays_afterEndTime = int(window_size / 2) -1
    else:
        extraTradeDays_afterEndTime = int(window_size / 2)
    #data_afterNTimesFilter_list = [close.values]
    tmp_close = close
    top_half_window = list(tmp_close)[:extraTradeDays_beforeStartTime]
    end_half_window = list(tmp_close)[-extraTradeDays_afterEndTime:]
    for i in range(filterTimes):
        # top_half_window = list(tmp_close)[:half_windowSize]
        # end_half_window = list(tmp_close)[-half_windowSize:]
        middle = list(data_oneTime_filter(tmp_close, window_size, window_beta))
        close_filtered_top_end = top_half_window + middle + end_half_window
        tmp_close = close_filtered_top_end
    #输出为n次滤波后的结果
    data_afterNTimesFilter = tmp_close #tmp_close[extraTradeDays_beforeStartTime:-extraTradeDays_afterEndTime]

    return  data_afterNTimesFilter
    
#输入参数init_dat是n次滤波后的结果，使用时先调用data_filter_n函数，将data_filter_n函数返回的结果作为init_dat
def tag_FilteredData_upDown(data_afterFilterNTimes):
        
    oneDayForward_list = data_afterFilterNTimes[1:]
    zeroDayForward_list = data_afterFilterNTimes[:-1]

    diff_list = [oneDay-zeroDay for oneDay, zeroDay in zip(oneDayForward_list,zeroDayForward_list)]

    tag_for_upDown = []

    for i in range(len(diff_list)):
        if i == 0:
            if diff_list[i] > 0:
                tag_for_upDown.append(trend_type['up'])
            elif diff_list[i] < 0:
                tag_for_upDown.append(trend_type['down'])
        elif i > 0:
            if diff_list[i] > 0:
                tag_for_upDown.append(trend_type['up'])
            elif diff_list[i] < 0:
                tag_for_upDown.append(trend_type['down'])
            else:
                if diff_list[i-1] < 0 and diff_list[i+1] > 0:
                    tag_for_upDown.append(trend_type['up'])
                elif diff_list[i - 1] < 0 and diff_list[i + 1] < 0:
                    tag_for_upDown.append(trend_type['down'])
                elif diff_list[i-1] > 0 and diff_list[i+1] < 0:
                    tag_for_upDown.append(trend_type['down'])
                elif diff_list[i-1] > 0 and diff_list[i+1] > 0:
                    tag_for_upDown.append(trend_type['up'])
                else:
                    tag_for_upDown.append(trend_type['up'])
    
    return  tag_for_upDown


# 输入参数，data_filter是函数data_filter_n的结果，tag_data_filter是函数tag_data_filter_n的结果，k_threshould是一个涨跌区间内，将当前状态定义为急和缓的阈值
def tag_sharpGentle(data_afterNTimesFilter, tag_for_upDown, slope_threshold):#data_afterNTimesFilter,tag_for_upDown
    change_loc = []
    change_loc.append(0)
    #slope_type = {'sharp': 5, 'gentle': 4}
    for i in range(1, len(tag_for_upDown)):
        if tag_for_upDown[i] == tag_for_upDown[i - 1]:
            pass
        elif tag_for_upDown[i] != tag_for_upDown[i - 1]:
            change_loc.append(i)
    if change_loc[-1] == len(tag_for_upDown) - 1:
        pass
    else:
        change_loc.append(len(tag_for_upDown) - 1)

    change_loc_values = []
    for loc in change_loc:
        change_loc_values.append(data_afterNTimesFilter[loc])

    k_s = []
    for i in range(len(change_loc) - 1):
        k = (change_loc_values[i + 1] - change_loc_values[i]) / (change_loc[i + 1] - change_loc[i])
        k_s.append(k)
    t_k = []
    for k in k_s:
        if abs(k) >= slope_threshold:
            t_k.append(slope_type['sharp'])
        else:
            t_k.append(slope_type['gentle'])

    tag_k = []
    for i in range(len(t_k)):
        a = t_k[i]
        for j in range(change_loc[i], change_loc[i + 1]):
            tag_k.append(a)
    tag_k.append(tag_k[-1])
    return tag_k

def tag_data_sharpGentleUpDown(close, window_size, window_beta, filterTimes_for_upDown, filterTimes_for_sharpGentle, slope_threshold):
    data_afterNTimesFilter = data_filter_nTimes(close, window_size, window_beta, filterTimes_for_upDown)
    ## there is no tag of 'upDown' for the last data
    tag_for_upDown = tag_FilteredData_upDown(data_afterNTimesFilter)
    ## there is no tag for the last data

    data_afterNTimesFilter_slope = data_filter_nTimes(close, window_size, window_beta, filterTimes_for_sharpGentle)
    tag_for_sharpGentle = tag_FilteredData_upDown(data_afterNTimesFilter_slope)

    tag_for_sharpGentle = tag_sharpGentle(data_afterNTimesFilter_slope, tag_for_upDown, slope_threshold=slope_threshold)

    # slope_type = {'sharp': 5, 'gentle': 4}
    # trend_type = {'up': 1, 'down': 0}
    tuple_list = [(slope_type['sharp'], trend_type['up']), (slope_type['sharp'], trend_type['down']),
                 (slope_type['gentle'], trend_type['up']), (slope_type['gentle'], trend_type['down'])]
    tag_tuple = zip(tag_for_sharpGentle,tag_for_upDown)
    tag = [tuple_list.index(v) for v in tag_tuple]
    return tag,data_afterNTimesFilter
    
def get_datasets_2(close,parameter_dict,show_label=True):
    """
    it has already cut the top NTradeDays, which is used for calculating the indicators.
    But the data it returns contain the extra days for tagging data.
    """
    window_size = parameter_dict['filter_windowSize']
    window_beta = parameter_dict['kaiser_beta']
    filterTimes_for_upDown = parameter_dict['filterTimes_for_upDown']
    filterTimes_for_sharpGentle = parameter_dict['filterTimes_for_sharpGentle']
    slope_threshold = parameter_dict['slope_threshold']
    extraTradeDays = parameter_dict['extraTradeDays_afterEndTime_for_filter']
    raw_y ,data_afterNTimesFilter = tag_data_sharpGentleUpDown(close, window_size=window_size, window_beta=window_beta,
                                                               filterTimes_for_upDown =filterTimes_for_upDown,
                                                               filterTimes_for_sharpGentle =filterTimes_for_sharpGentle, slope_threshold=slope_threshold)
    raw_y = np.array(raw_y)
    close = np.array(close.tolist())
    pct = np.diff(close)/close[:-1]
    
    raw_x = get_indicators(pct)
   # print 'filtered_close',len(filtered_close),filtered_close[:10]
    
    nan_num = max(np.where(np.isnan(raw_x))[0])+1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    if NTradeDays_for_indicatorCalculation<nan_num+1:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN+1,i.e.%s'%str(nan_num+1))

    x = raw_x[NTradeDays_for_indicatorCalculation-1:-1]
    labels = raw_y[NTradeDays_for_indicatorCalculation:]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-1]
    filter_data_for_use = data_afterNTimesFilter[NTradeDays_for_indicatorCalculation:-1]

    extraTradeDays = extraTradeDays - 1
    x, labels = x[:-extraTradeDays], labels[:-extraTradeDays]
    filter_data_for_use, close_for_use = filter_data_for_use[:-extraTradeDays], close_for_use[:-extraTradeDays]
    return x,labels,close_for_use,filter_data_for_use


def get_x_y_sharpGentleUpDown(close, parameter_dict):
    """
        the variables it returns has cut the extraTradeDays for indicator in the top positions and also cut the
        extraTradeDays for tagging data in the last positions. The data it returns is exactly the same size as the
        data between startTime and endTime
    """
    window_size = parameter_dict['filter_windowSize']
    window_beta = parameter_dict['kaiser_beta']
    filterTimes_for_upDown = parameter_dict['filterTimes_for_upDown']
    filterTimes_for_sharpGentle = parameter_dict['filterTimes_for_sharpGentle']
    slope_threshold = parameter_dict['slope_threshold']
    extraTradeDays = int(window_size/2)
    ## there is no tag of 'upDown' for the last data and there is no data lost when filter data, so
    ## 'data_afterNTimesFilter' has one more data than 'raw_y'
    raw_y, data_afterNTimesFilter = tag_data_sharpGentleUpDown(close, window_size=window_size, window_beta=window_beta,
                                                               filterTimes_for_upDown=filterTimes_for_upDown,
                                                               filterTimes_for_sharpGentle=filterTimes_for_sharpGentle,
                                                               slope_threshold=slope_threshold)
    raw_y = np.array(raw_y)
    close = np.array(close.tolist())
    return_rate = np.diff(close) / close[:-1]
    ### calculate indicators with log_return, other than 'close' price
    raw_x = get_indicators(return_rate)
    # the number of max length of 'nan' in indicators
    nan_num = max(np.where(np.isnan(raw_x))[0]) + 1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    if NTradeDays_for_indicatorCalculation < nan_num:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN,i.e.%s' % str(nan_num ))
    # the return_rate of the first day in 'close' can not be calculated, so there is one day miss in the first position
    #  in 'close_return_rate' when comparing with 'close'. When cut 'NTradeDays_for_indicatorCalculation' days in 'close',
    #  we only need to cut 'NTradeDays_for_indicatorCalculation-1' days in 'raw_x'
    # 'raw_label_y' is the label array for 'close'. Each day's label needs the next day's close price to tag it,
    # so there is no label for the last day in 'close'. We need to cut one day in the last of 'close' for the alignment
    # with 'labels'. So does 'filter_data_for_use' and 'raw_x_indicators'.
    x = raw_x[NTradeDays_for_indicatorCalculation - 1:-1]
    labels = raw_y[NTradeDays_for_indicatorCalculation :]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-1]
    filter_data_for_use = data_afterNTimesFilter[NTradeDays_for_indicatorCalculation:-1]

    # original 'close' has extraTradeDays after 'endTime'. We have cut one day because of 'labels', so we just need to
    # cut 'extraTradeDays-1' days to make the data align with data between startTime and endTime.
    extraTradeDays = extraTradeDays-1
    x, labels = x[:-extraTradeDays], labels[:-extraTradeDays]
    filter_data_for_use, close_for_use = filter_data_for_use[:-extraTradeDays], close_for_use[:-extraTradeDays]
    return x, labels, filter_data_for_use, close_for_use

def get_tag_bottomTopUpDown(close, parameter_dict):
    window_size = parameter_dict['filter_windowSize']
    window_beta = parameter_dict['kaiser_beta']
    filterTimes = parameter_dict['filterTimes_for_upDown']
    #filtered_data = data_oneTime_filter(close, window_size, window_beta)
    filtered_data = data_filter_nTimes(close, window_size, window_beta, filterTimes)

    tag= get_4labels_bottomTopUpDown(filtered_data, parameter_dict)
    return tag, np.array(filtered_data)


def get_4labels_bottomTopUpDown(filtered_data, parameter_dict):
    knee_idx = [0]
    filtered_data = np.array(filtered_data)
    data_len = filtered_data.shape[0]
    for i in range(1,data_len):
        if i == data_len-1:
            knee_idx.append(i)
        else:
            if (filtered_data[i+1]-filtered_data[i])*(filtered_data[i]-filtered_data[i-1])<0:
                knee_idx.append(i)
            else:
                pass

    knee_close = filtered_data[knee_idx]
    change_percent = parameter_dict['change_percent']
    break_t, break_p = get3d.break_point(change_percent, knee_idx, knee_close)
    if not break_t[-1] == knee_idx[-1]:
        break_t.append(knee_idx[-1])
        break_p.append(knee_close[-1])

    break_idx = np.array(break_t)
    kneeNum_at_bottomTop = parameter_dict['kneeNum_at_bottomTop']
    labels = np.zeros(data_len)
    label_type = {'bottom':0, 'up':1,'top':2, 'down':3}
    if filtered_data[break_idx[1]] - filtered_data[break_idx[0]] > 0:
        for i in range(len(break_idx)-1):

                if i%2==0:
                    labels[break_idx[i]:break_idx[i]+kneeNum_at_bottomTop] = label_type['bottom']
                    labels[break_idx[i]+kneeNum_at_bottomTop:break_idx[i+1]-kneeNum_at_bottomTop] = label_type['up']
                    labels[break_idx[i+1]-kneeNum_at_bottomTop:break_idx[i+1]]= label_type['top']
                else:
                    labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['top']
                    labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['down']
                    labels[break_idx[i+1]-kneeNum_at_bottomTop:break_idx[i+1]]=label_type['bottom']
    else:#filtered_data[break_idx[1]] - filtered_data[break_idx[0]] < 0:
        for i in range(len(break_idx) - 1):
            if i % 2 == 0:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['top']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['down']
                labels[break_idx[i+1]-kneeNum_at_bottomTop:break_idx[i+1]] = label_type['bottom']
            else:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['bottom']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['up']
                labels[break_idx[i+1]-kneeNum_at_bottomTop:break_idx[i+1]] = label_type['top']

    return labels

def get_2labels_upDown(filtered_data, parameter_dict):
    knee_idx = [0]
    filtered_data = np.array(filtered_data)
    data_len = filtered_data.shape[0]
    for i in range(1, data_len):
        if i == data_len - 1:
            knee_idx.append(i)
        else:
            if (filtered_data[i + 1] - filtered_data[i]) * (filtered_data[i] - filtered_data[i - 1]) < 0:
                knee_idx.append(i)
            else:
                pass

    knee_close = filtered_data[knee_idx]
    change_percent = parameter_dict['change_percent']
    break_t, break_p = get3d.break_point(change_percent, knee_idx, knee_close)
    if not break_t[-1] == knee_idx[-1]:
        break_t.append(knee_idx[-1])
        break_p.append(knee_close[-1])

    break_idx = np.array(break_t)
    kneeNum_at_bottomTop = parameter_dict['kneeNum_at_bottomTop']
    labels = np.zeros(data_len)
    label_type = {'bottom': 0, 'up': 1, 'top': 2, 'down': 3}
    if filtered_data[break_idx[1]] - filtered_data[break_idx[0]] > 0:
        for i in range(len(break_idx) - 1):

            if i % 2 == 0:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['up']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['up']
                labels[break_idx[i + 1] - kneeNum_at_bottomTop:break_idx[i + 1]] = label_type['down']
            else:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['down']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['down']
                labels[break_idx[i + 1] - kneeNum_at_bottomTop:break_idx[i + 1]] = label_type['up']
    else:  # filtered_data[break_idx[1]] - filtered_data[break_idx[0]] < 0:
        for i in range(len(break_idx) - 1):
            if i % 2 == 0:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['down']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['down']
                labels[break_idx[i + 1] - kneeNum_at_bottomTop:break_idx[i + 1]] = label_type['up']
            else:
                labels[break_idx[i]:break_idx[i] + kneeNum_at_bottomTop] = label_type['up']
                labels[break_idx[i] + kneeNum_at_bottomTop:break_idx[i + 1] - kneeNum_at_bottomTop] = label_type['up']
                labels[break_idx[i + 1] - kneeNum_at_bottomTop:break_idx[i + 1]] = label_type['down']

    return labels
def get_x_y_bottomTopUpDown(close, parameter_dict):#BottomTopUpDown
    """ """
    raw_y, filtered_data = get_tag_bottomTopUpDown(close, parameter_dict)

    raw_y = np.array(raw_y)
    close = np.array(close.tolist())
    return_rate = np.diff(close) / close[:-1]
    raw_x = get_indicators(close[1:],return_rate)
    # print 'filtered_close',len(filtered_close),filtered_close[:10]
    nan_num = max(np.where(np.isnan(raw_x))[0]) + 1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    extraTradeDays_afterEndTime = parameter_dict['extraTradeDays_afterEndTime']
    if NTradeDays_for_indicatorCalculation < nan_num:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN+1,i.e.%s' % str(nan_num ))

    x = raw_x[NTradeDays_for_indicatorCalculation-1:-extraTradeDays_afterEndTime]
    labels = raw_y[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    filter_data_for_use = filtered_data[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    print('x,y,close,filtered_close',x.shape,labels.shape,close_for_use.shape, filter_data_for_use.shape)

    return x, labels.astype(np.int), filter_data_for_use, close_for_use

def get_x_y_bottomTopUpDown2(close, parameter_dict):#BottomTopUpDown
    """ """
    raw_y, filtered_data = get_tag_bottomTopUpDown(close, parameter_dict)

    raw_y = np.array(raw_y)
    close = np.array(close.tolist())
    #return_rate = np.diff(close) / close[:-1]
    raw_x = get_indicators(close)
    # print 'filtered_close',len(filtered_close),filtered_close[:10]
    nan_num = max(np.where(np.isnan(raw_x))[0]) + 1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    extraTradeDays_afterEndTime = parameter_dict['extraTradeDays_afterEndTime']
    if NTradeDays_for_indicatorCalculation < nan_num:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN+1,i.e.%s' % str(nan_num ))


    x = raw_x[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    labels = raw_y[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    filter_data_for_use = filtered_data[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    print('x,y,close,filtered_close',x.shape,labels.shape,close_for_use.shape, filter_data_for_use.shape)

    return x, labels.astype(np.int), filter_data_for_use, close_for_use


def get_balanced_datasets(x,y,parameter_dict):
    nb_class = parameter_dict['nb_classes']
    print 'nb_classes',nb_class
    
    indexArrayOfEachClass_list = [np.where(y==i)[0] for i in range(nb_class)]
    min_len = min([v.shape[0] for v in indexArrayOfEachClass_list])
    print 'the minimum data number in %s classes'%nb_class,min_len

    selected_idx_list = [np.random.choice(range(len(indexArrayOfOneClass)),min_len).tolist() for indexArrayOfOneClass in indexArrayOfEachClass_list]
    balanced_x_list = list(map(lambda idx_,select_id: x[idx_[select_id],:], indexArrayOfEachClass_list,selected_idx_list))
    balanced_y_list = list(map(lambda idx_,select_id:y[idx_[select_id]], indexArrayOfEachClass_list,selected_idx_list))
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
    

if __name__=='__main__':
   pass
    
    
    
    

    
    
    
    
    
    
    
    

    
        