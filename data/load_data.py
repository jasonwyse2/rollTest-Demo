#coding:utf-8
import pandas as pd
import numpy as np
import scipy.signal as signal
import datalib.util.get_3d_data_set as get3d
from matplotlib import pyplot
import tool
from indicator_config import get_indicator_handle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

slope_type = {'sharp':5,'gentle':4}
trend_type = {'up': 1, 'down': 0}
day_field='date, open, high, low, close, pct_chg, volume, amt '
minute_field='time, open, high, low, close, pct_chg, volume, amt '

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

    tmp_close = close
    top_half_window = list(tmp_close)[:extraTradeDays_beforeStartTime]
    end_half_window = list(tmp_close)[-extraTradeDays_afterEndTime:]
    for i in range(filterTimes):
        middle = list(data_oneTime_filter(tmp_close, window_size, window_beta))
        close_filtered_top_end = top_half_window + middle + end_half_window
        tmp_close = close_filtered_top_end

    data_afterNTimesFilter = tmp_close
    return  np.array(data_afterNTimesFilter)

def tag_upDown(data_afterFilterNTimes):
        
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

def get_x_y_repeat(raw_data_mat, data_parameters_dict):
    show_label = data_parameters_dict['show_label']
    closeSeries_num = raw_data_mat.shape[1]
    x_list, y_list = [], []
    filtered_close_list, close_list = [], []
    for i in range(closeSeries_num):
        close_array = raw_data_mat[:, i]
        close_array = close_array[close_array>0]
        #close = pd.Series(close_array)
        raw_data_df = pd.DataFrame(close_array)
        #x_price_list = tool.get_xPriceList(raw_data_mat, data_parameters_dict)
        [x, y, filtered_close_for_use, close_for_use] = get_x_y(raw_data_df, data_parameters_dict)
        if show_label == True:
            tool.show_fig(y, filtered_close_for_use, close_for_use)
        x_list.append(x)
        y_list.append(y)
        filtered_close_list.append(filtered_close_for_use)
        close_list.append(close_for_use)
    return [x_list, y_list, filtered_close_list, close_list]

def get_x_y(raw_data_df, parameter_dict):
    fourlabelType = parameter_dict['taskType']
    if fourlabelType == 'SharpGentleUpDown':
        x, y, filtered_close_for_use, close_for_use = get_x_y_sharpGentleUpDown(raw_data_df, parameter_dict)
    elif fourlabelType == 'BottomTopUpDown':
        x, y, filtered_close_for_use, close_for_use = get_x_y_bottomTopUpDown(raw_data_df, parameter_dict)
    elif fourlabelType == 'Volatility':
        x, y, filtered_close_for_use, close_for_use = get_x_y_volatility(raw_data_df, parameter_dict)

    return [x, y, filtered_close_for_use, close_for_use]

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
    tag_for_upDown = tag_upDown(data_afterNTimesFilter)
    ## there is no tag for the last data

    data_afterNTimesFilter_slope = data_filter_nTimes(close, window_size, window_beta, filterTimes_for_sharpGentle)
    tag_for_sharpGentle = tag_upDown(data_afterNTimesFilter_slope)

    tag_for_sharpGentle = tag_sharpGentle(data_afterNTimesFilter_slope, tag_for_upDown, slope_threshold=slope_threshold)

    # slope_type = {'sharp': 5, 'gentle': 4}
    # trend_type = {'up': 1, 'down': 0}
    tuple_list = [(slope_type['sharp'], trend_type['up']), (slope_type['sharp'], trend_type['down']),
                 (slope_type['gentle'], trend_type['up']), (slope_type['gentle'], trend_type['down'])]
    tag_tuple = zip(tag_for_sharpGentle,tag_for_upDown)
    tag = [tuple_list.index(v) for v in tag_tuple]
    return tag,data_afterNTimesFilter
    
def get_x_y_sharpGentleUpDown(raw_data_df, parameter_dict):
    """
        the variables it returns has cut the extraTradeDays for indicator in the top positions and also cut the
        extraTradeDays for tagging data in the last positions. The data it returns is exactly the same size as the
        data between startTime and endTime
    """
    dayOrMinute = parameter_dict['dayOrMinute']
    if dayOrMinute == 'minute_no_simulative':
        close = np.array(raw_data_df['close'])
    else:
        close = np.array(raw_data_df.iloc[:,0])
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
    #close = np.array(close.tolist())
    #return_rate = np.diff(close) / close[:-1]
    ### calculate indicators with log_return, other than 'close' price
    indicator_combination = parameter_dict['indicator_combination']
    get_indicators_handle = get_indicator_handle(indicator_combination)
    raw_x = get_indicators_handle(raw_data_df)
    #raw_x = get_indicators(close, return_rate)
    # the number of max length of 'nan' in indicators
    Nan_num = max(np.where(np.isnan(raw_x))[0]) + 1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    if NTradeDays_for_indicatorCalculation < Nan_num:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN,i.e.%s' % str(Nan_num ))
    # the return_rate of the first day in 'close' can not be calculated, so there is one day miss in the first position
    #  in 'close_return_rate' when comparing with 'close'. When cut 'NTradeDays_for_indicatorCalculation' days in 'close',
    #  we only need to cut 'NTradeDays_for_indicatorCalculation-1' days in 'raw_x'
    # 'raw_label_y' is the label array for 'close'. Each day's label needs the next day's close price to tag it,
    # so there is no label for the last day in 'close'. We need to cut one day in the last of 'close' for the alignment
    # with 'labels'. So does 'filter_data_for_use' and 'raw_x_indicators'.
    x = raw_x[NTradeDays_for_indicatorCalculation - 1:-1]
    y = raw_y[NTradeDays_for_indicatorCalculation :]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-1]
    filter_data_for_use = data_afterNTimesFilter[NTradeDays_for_indicatorCalculation:-1]

    # original 'close' has extraTradeDays after 'endTime'. We have cut one day because of 'labels', so we just need to
    # cut 'extraTradeDays-1' days to make the data align with data between startTime and endTime.
    extraTradeDays = extraTradeDays-1
    x, y = x[:-extraTradeDays], y[:-extraTradeDays]
    filter_data_for_use, close_for_use = filter_data_for_use[:-extraTradeDays], close_for_use[:-extraTradeDays]
    print('x,y,close,filtered_close', x.shape, y.shape, close_for_use.shape, filter_data_for_use.shape)
    return x, y, filter_data_for_use, close_for_use

def get_tag_volatility(close, parameter_dict):
    volatility_window = parameter_dict['volatility_window']
    global_parameter_dict = parameter_dict['global_parameter_dict']
    close_volatility = []
    for i in range(len(close)):
        close_range = close[i-volatility_window:i+volatility_window]
        close_volatility.append(np.var(close_range))
    # pyplot.hist(tag,bins=100)
    # pyplot.show()
    # remove 'NaN' value
    close_volatility = close_volatility[volatility_window:-volatility_window]
    close_volatility = np.array(close_volatility)
    # figure1 = plt.figure()
    # ax1 = figure1.add_subplot(211)
    # ax2 = figure1.add_subplot(212)
    # n, bins, patchs = ax1.hist(close_volatility, bins=1000, color='red', normed=1, cumulative=True)
    # # plt.plot(train_y_ndarray)
    # # plt.xlim(0,10000)
    # n, bins, patchs = ax2.hist(close_volatility, bins=1000, color='red', normed=0, )

    nb_classes = parameter_dict['nb_classes']
    estimator = KMeans(n_clusters=nb_classes)
    estimator.fit(close_volatility.reshape(-1, 1))
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_

    labels = list(set(label_pred))
    last_element_list = []
    clusters = []
    idx_clusters = []
    for i in range(len(labels)):
        label_i_idx = label_pred == labels[i]
        tmp = close_volatility[label_i_idx]
        tmp = np.sort(tmp)
        # print tmp[-1]
        last_element_list.append(tmp[-1])
        clusters.append(tmp.tolist())
        idx_clusters.append(label_i_idx)

    global_parameter_dict['last_element_list']= last_element_list
    idx_array = np.argsort(np.array(last_element_list))
    clusters_elementSorted_list = []
    idx_clusters_list = []
    lastElementOfCluster_byOrder_list = []
    for i in range(len(idx_array)):
        clusters_elementSorted_list.append(clusters[idx_array[i]])
        idx_clusters_list.append(idx_clusters[idx_array[i]])
        lastElementOfCluster_byOrder_list.append(clusters[idx_array[i]][-1])
        # print clusters[idx_array[i]][0:10]
        # print np.sort(train_y_ndarray[idx_clusters[idx_array[i]]])[0:10]
    global_parameter_dict['lastElementOfCluster_byOrder_list'] = lastElementOfCluster_byOrder_list
    train_y_ndarray = label_pred  # initialize real label
    for i in range(len(idx_array)):
        train_y_ndarray[idx_clusters_list[i]] = i

    train_y_ndarray1 = np.concatenate((np.zeros(volatility_window), train_y_ndarray,np.zeros(volatility_window)),axis=0).astype(np.int32)
    #add 'Nan' value

    return train_y_ndarray1

def get_tag_bottomTopUpDown(close, parameter_dict):
    window_size = parameter_dict['filter_windowSize']
    window_beta = parameter_dict['kaiser_beta']
    filterTimes = parameter_dict['filterTimes_for_upDown']
    #filtered_data = data_oneTime_filter(close, window_size, window_beta)
    filtered_data = data_filter_nTimes(close, window_size, window_beta, filterTimes)

    tag= get_4labels_bottomTopUpDown(filtered_data, parameter_dict)
    return tag, np.array(filtered_data)

def get_4labels_bottomTopUpDown_backup(filtered_data, parameter_dict):
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
    change_threshold = parameter_dict['change_threshold']
    break_t, break_p = get3d.break_point(change_threshold, knee_idx, knee_close)
    if not break_t[-1] == knee_idx[-1]:
        break_t.append(knee_idx[-1])
        break_p.append(knee_close[-1])

    break_idx = np.array(break_t)
    kneeNum_at_bottomTop = parameter_dict['kneeNum_at_bottomTop']
    labels = np.zeros(data_len)
    label_type = {'bottom':0, 'up':1,'top':2, 'down':3}
    pct_threshold = 0.003
    init_tupe = 1,1,0,1

    if filtered_data[break_idx[1]] - filtered_data[break_idx[0]] > 0:
        for i in range(len(break_idx)-1):

                if i%2==0:
                    right_bottum,right_top,left_bottum,left_top = init_tupe
                    for j in range(1,kneeNum_at_bottomTop):
                        if (filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]]) /filtered_data[break_idx[i]] < pct_threshold:
                            right_bottum += 1
                        else:
                            break

                    for j in range(1, kneeNum_at_bottomTop):
                        if (filtered_data[break_idx[i+1]] - filtered_data[break_idx[i+1]-j]) /filtered_data[break_idx[i+1]-j] < pct_threshold:
                            left_top = +1
                        else:
                            break

                    labels[break_idx[i]:break_idx[i]+right_bottum] = label_type['bottom']
                    labels[break_idx[i]+right_bottum:break_idx[i+1]-left_top] = label_type['up']
                    labels[break_idx[i+1]-left_top:break_idx[i+1]]= label_type['top']
                else:
                    right_bottum, right_top, left_bottum, left_top = init_tupe
                    for j in range(1, kneeNum_at_bottomTop):
                        if abs((filtered_data[break_idx[i]+j] - filtered_data[break_idx[i]])) / filtered_data[break_idx[i]] < pct_threshold:
                            right_top += 1
                        else:
                            break
                    for j in range(1, kneeNum_at_bottomTop):
                        if abs((filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j])) / filtered_data[break_idx[i+1] - j] < pct_threshold:
                            left_bottum += 1
                        else:
                            break

                    labels[break_idx[i]:break_idx[i] + right_top] = label_type['top']
                    labels[break_idx[i] + right_top:break_idx[i + 1] - left_bottum] = label_type['down']
                    labels[break_idx[i+1]-left_bottum:break_idx[i+1]]=label_type['bottom']
    else:#filtered_data[break_idx[1]] - filtered_data[break_idx[0]] < 0:

        for i in range(len(break_idx) - 1):

            if i % 2 == 0:
                right_bottum, right_top, left_bottum, left_top = init_tupe

                for j in range(1, kneeNum_at_bottomTop):
                    if abs((filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]])) / filtered_data[break_idx[i]] < pct_threshold:
                        right_top += 1
                    else:
                        break
                for j in range(1, kneeNum_at_bottomTop):
                    if abs((filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j])) / filtered_data[break_idx[i+1] - j] < pct_threshold:
                        left_bottum += 1
                    else:
                        break

                labels[break_idx[i]:break_idx[i] + right_top] = label_type['top']
                labels[break_idx[i] + right_top:break_idx[i + 1] - left_bottum] = label_type['down']
                labels[break_idx[i + 1] - left_bottum:break_idx[i + 1]] = label_type['bottom']

            else:

                right_bottum, right_top, left_bottum, left_top = init_tupe
                for j in range(1, kneeNum_at_bottomTop):
                    if (filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]]) / filtered_data[break_idx[i]] < pct_threshold:
                        right_bottum += 1
                    else:
                        break
                for j in range(1, kneeNum_at_bottomTop):
                    if (filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j]) / filtered_data[break_idx[i+1] - j] < pct_threshold:
                        left_top += 1
                    else:
                       break

                labels[break_idx[i]:break_idx[i] + right_bottum] = label_type['bottom']
                labels[break_idx[i] + right_bottum:break_idx[i + 1] - left_top] = label_type['up']
                labels[break_idx[i + 1] - left_top:break_idx[i + 1]] = label_type['top']

    return labels

def get_4labels_bottomTopUpDown_abslute(filtered_data, parameter_dict):
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
    change_threshold = parameter_dict['change_threshold']
    break_t, break_p = get3d.break_point(change_threshold, knee_idx, knee_close)
    if not break_t[-1] == knee_idx[-1]:
        break_t.append(knee_idx[-1])
        break_p.append(knee_close[-1])


    break_idx = np.array(break_t)
    kneeNum_at_bottomTop = parameter_dict['kneeNum_at_bottomTop']
    labels = np.zeros(data_len)
    label_type = {'bottom':0, 'up':1,'top':2, 'down':3}
    pct_threshold = 90
    init_tupe = 0,0,0,0
    if filtered_data[break_idx[1]] - filtered_data[break_idx[0]] > 0:
        for i in range(len(break_idx)-1):

                if i%2==0:
                    right_bottum,right_top,left_bottum,left_top = init_tupe
                    for j in range(1,kneeNum_at_bottomTop):
                        if (filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]])  < pct_threshold:
                            right_bottum += 1
                        else:
                            break

                    for j in range(1, kneeNum_at_bottomTop):
                        if (filtered_data[break_idx[i+1]] - filtered_data[break_idx[i+1]-j])  < pct_threshold:
                            left_top += 1
                        else:
                            break

                    labels[break_idx[i]:break_idx[i]+right_bottum] = label_type['bottom']
                    labels[break_idx[i]+right_bottum:break_idx[i+1]-left_top] = label_type['up']
                    labels[break_idx[i+1]-left_top:break_idx[i+1]]= label_type['top']
                else:
                    right_bottum, right_top, left_bottum, left_top = init_tupe
                    for j in range(1, kneeNum_at_bottomTop):
                        if abs((filtered_data[break_idx[i]+j] - filtered_data[break_idx[i]])) < pct_threshold:
                            right_top += 1
                        else:
                            break

                    for j in range(1, kneeNum_at_bottomTop):
                        if abs((filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j])) < pct_threshold:
                            left_bottum += 1
                        else:
                            break

                    labels[break_idx[i]:break_idx[i] + right_top] = label_type['top']
                    labels[break_idx[i] + right_top:break_idx[i + 1] - left_bottum] = label_type['down']
                    labels[break_idx[i+1]-left_bottum:break_idx[i+1]]=label_type['bottom']
    else:#filtered_data[break_idx[1]] - filtered_data[break_idx[0]] < 0:

        for i in range(len(break_idx) - 1):

            if i % 2 == 0:
                right_bottum, right_top, left_bottum, left_top = init_tupe

                for j in range(1, kneeNum_at_bottomTop):
                    if abs((filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]])) < pct_threshold:
                        right_top += 1
                    else:
                        break

                for j in range(1, kneeNum_at_bottomTop):
                    if abs((filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j])) < pct_threshold:
                        left_bottum += 1
                    else:
                        break

                labels[break_idx[i]:break_idx[i] + right_top] = label_type['top']
                labels[break_idx[i] + right_top:break_idx[i + 1] - left_bottum] = label_type['down']
                labels[break_idx[i + 1] - left_bottum:break_idx[i + 1]] = label_type['bottom']

            else:

                right_bottum, right_top, left_bottum, left_top = init_tupe
                for j in range(1, kneeNum_at_bottomTop):
                    if (filtered_data[break_idx[i] + j] - filtered_data[break_idx[i]]) < pct_threshold:
                        right_bottum += 1
                    else:
                        break

                for j in range(1, kneeNum_at_bottomTop):
                    if (filtered_data[break_idx[i + 1]] - filtered_data[break_idx[i+1] - j])  < pct_threshold:
                        left_top += 1
                    else:
                       break

                labels[break_idx[i]:break_idx[i] + right_bottum] = label_type['bottom']
                labels[break_idx[i] + right_bottum:break_idx[i + 1] - left_top] = label_type['up']
                labels[break_idx[i + 1] - left_top:break_idx[i + 1]] = label_type['top']

    return labels

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
    change_threshold = parameter_dict['change_threshold']
    break_t, break_p = break_point(change_threshold, knee_idx, knee_close)
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
def break_point(change_percent, knee_idx_list, knee_close_list):
    break_close = [knee_close_list[0]]
    break_close_index = [knee_idx_list[0]]
    i = 1
    tmp = -1
    trend_type = {'up_trend':1,'down_trend':-1}
    while i < len(knee_close_list[:-1]):
        # print i

        if i - tmp == 0:
            break
        tmp = i
        if knee_close_list[i] - break_close[-1] > 0:
            flag = trend_type['up_trend']
            tmp_high = knee_close_list[i]
            tmp_high_index = knee_idx_list[i]
        else:
            flag = trend_type['down_trend']
            tmp_low = knee_close_list[i]
            tmp_low_index = knee_idx_list[i]

        if flag == trend_type['up_trend']:
            for j, v in enumerate(knee_close_list[i + 1:]):
                if (v - tmp_high) / tmp_high < -change_percent:
                    break_close.append(tmp_high)
                    break_close_index.append(tmp_high_index)
                    i += j + 1
                    break
                else:
                    if v > tmp_high:
                        tmp_high = v
                        tmp_high_index = knee_idx_list[i + j + 1]

                    else:
                        continue
        else:#flag == trend_type['down_trend']
            for j, v in enumerate(knee_close_list[i + 1:]):
                if (v - tmp_low) / tmp_low > change_percent:
                    break_close.append(tmp_low)
                    break_close_index.append(tmp_low_index)
                    i += j + 1
                    break
                else:
                    if v < tmp_low:
                        tmp_low = v
                        tmp_low_index = knee_idx_list[i + j + 1]

                    else:
                        continue
    return break_close_index, break_close

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

def get_x_y_bottomTopUpDown(raw_data_df, parameter_dict):#BottomTopUpDown
    """ """
    dayOrMinute = parameter_dict['dayOrMinute']
    minuteType_dict = parameter_dict['minuteType_dict']
    if dayOrMinute == minuteType_dict['minuteNoSimulative']:
        close = np.array(raw_data_df['close'])
    else:
        close = np.array(raw_data_df.iloc[:,0])

    raw_y, filtered_data = get_tag_bottomTopUpDown(close, parameter_dict)
    raw_y = np.array(raw_y)

    indicator_combination = parameter_dict['indicator_combination']
    get_indicators_handle = get_indicator_handle(indicator_combination)
    raw_x = get_indicators_handle(raw_data_df)

    # print 'filtered_close',len(filtered_close),filtered_close[:10]
    Nan_num = max(np.where(np.isnan(raw_x))[0]) + 1
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    extraTradeDays_afterEndTime = parameter_dict['extraTradeDays_afterEndTime']
    if NTradeDays_for_indicatorCalculation < Nan_num:
        raise Exception('"NTradeDays_for_indicatorCalculation" can not be less than NaN,i.e.%s' % str(Nan_num ))

    x = raw_x[NTradeDays_for_indicatorCalculation-1:-extraTradeDays_afterEndTime]
    labels = raw_y[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    filter_data_for_use = filtered_data[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    print('x,y,close,filtered_close',x.shape,labels.shape,close_for_use.shape, filter_data_for_use.shape)

    return x, labels.astype(np.int), filter_data_for_use, close_for_use
def get_x_y_volatility(raw_data_df, parameter_dict):
    dayOrMinute = parameter_dict['dayOrMinute']
    minuteType_dict = parameter_dict['minuteType_dict']
    dataType = parameter_dict['dataType']
    if dayOrMinute == minuteType_dict['minuteNoSimulative']:
        close = np.array(raw_data_df['close'])
    else:
        close = np.array(raw_data_df.iloc[:, 0])
    # close = np.diff(close) / close[:-1]
    if dataType=='train':
        raw_y = get_tag_volatility(close, parameter_dict)
    else:
        raw_y = get_tag_volatility_by_clusterRange(close, parameter_dict)

    indicator_combination = parameter_dict['indicator_combination']
    get_indicators_handle = get_indicator_handle(indicator_combination)
    raw_x = get_indicators_handle(raw_data_df)

    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    extraTradeDays_afterEndTime = parameter_dict['extraTradeDays_afterEndTime']

    x = raw_x[NTradeDays_for_indicatorCalculation - 1:-extraTradeDays_afterEndTime]
    labels = raw_y[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    close_for_use = close[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]
    filter_data_for_use = close[NTradeDays_for_indicatorCalculation:-extraTradeDays_afterEndTime]

    return x, labels.astype(np.int), filter_data_for_use, close_for_use
def get_tag_volatility_by_clusterRange(close, parameter_dict):
    volatility_window = parameter_dict['volatility_window']
    global_parameter_dict = parameter_dict['global_parameter_dict']
    lastElementOfCluster_byOrder_list=global_parameter_dict['lastElementOfCluster_byOrder_list']
    close_volatility = []
    for i in range(len(close)):
        close_range = close[i - volatility_window:i + volatility_window]
        close_volatility.append(np.var(close_range))

    close_volatility = close_volatility[volatility_window:-volatility_window]
    # initialize label array
    train_y_ndarray = close_volatility
    for i in range(len(close_volatility)):
        for j in range(len(lastElementOfCluster_byOrder_list)):
            if close_volatility[i]<=lastElementOfCluster_byOrder_list[j]:
                train_y_ndarray[i] = j
                break

    train_y_ndarray1 = np.concatenate((np.zeros(volatility_window), train_y_ndarray, np.zeros(volatility_window)),
                                      axis=0).astype(np.int32)

    return train_y_ndarray1
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
    
    
    
    

    
    
    
    
    
    
    
    

    
        