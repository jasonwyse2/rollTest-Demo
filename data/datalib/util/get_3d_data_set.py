# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import numpy as np
import talib


def extend_mat(k_data, n_kdata, N):
    ''' extend the scaled data which scaled by N minutes, extending the data everyday'''
    N_KData = np.zeros((k_data.shape[0], n_kdata.shape[1]))
    for k in range(n_kdata.shape[1]):
        for i in range(k_data.shape[0] / 241):
            for j in range(int(241 / N)):
                if j == int(241 / N) - 1:
                    N_KData[j * N + i * 241:, k] = n_kdata[j + i * 240 / N, k]
                else:
                    N_KData[j * N + i * 241:j * N + i * 241 + N, k] = n_kdata[j + i * 240 / N, k]

    return N_KData


def get_N_KData(n, k_data):
    '''get the 4P1V K_Data  with N_minutes as period'''
    num_day = k_data.shape[0] / 241
    num_n_everyday = 241 / n
    num_n_time = num_day * num_n_everyday
    n_kdata = np.zeros((num_n_time, k_data.shape[1]))
    if n == 240:
        for i in range(k_data.shape[0] / 241):
            for j in range(int(241 / n)):
                open_n = k_data[(j * n + i * 241):(j * n + i * 241) + n+1, :][0, 0]
                close_n = k_data[(j * n + i * 241):(j * n + i * 241) + n+1, :][-1, 3]
                high_n = np.max(k_data[(j * n + i * 241):(j * n + i * 241) + n+1, :4])
                low_n = np.min(k_data[(j * n + i * 241):(j * n + i * 241) + n+1, :4])
                volume_n = np.sum(k_data[(j * n + i * 241):(j * n + i * 241) + n+1, 4])
                n_kdata[j + i * 240 / n, 0] = open_n
                n_kdata[j + i * 240 / n, 1] = high_n
                n_kdata[j + i * 240 / n, 2] = low_n
                n_kdata[j + i * 240 / n, 3] = close_n
                n_kdata[j + i * 240 / n, 4] = volume_n
    else:
        for i in range(k_data.shape[0] / 241):
            for j in range(int(241 / n)):
                if j == int(241 / n) - 1:
                    open_n = k_data[(j * n + i * 241):(j * n + i * 241)+n+1, :][0, 0]
                    close_n = k_data[(j * n + i * 241):(j * n + i * 241)+n+1, :][-1, 3]
                    high_n = np.max(k_data[(j * n + i * 241):(j * n + i * 241)+n+1, :4])
                    low_n = np.min(k_data[(j * n + i * 241):(j * n + i * 241)+n+1, :4])
                    volume_n = np.sum(k_data[(j * n + i * 241):(j * n + i * 241)+n+1, 4])
                    n_kdata[j + i * 240 / n, 0] = open_n
                    n_kdata[j + i * 240 / n, 1] = high_n
                    n_kdata[j + i * 240 / n, 2] = low_n
                    n_kdata[j + i * 240 / n, 3] = close_n
                    n_kdata[j + i * 240 / n, 4] = volume_n

                else:
                    open_n = k_data[(j * n + i * 241):(j * n + i * 241) + n, :][0, 0]
                    close_n = k_data[(j * n + i * 241):(j * n + i * 241) + n, :][-1, 3]
                    high_n = np.max(k_data[(j * n + i * 241):(j * n + i * 241) + n, :4])
                    low_n = np.min(k_data[(j * n + i * 241):(j * n + i * 241) + n, :4])
                    volume_n = np.sum(k_data[(j * n + i * 241):(j * n + i * 241) + n, 4])
                    n_kdata[j + i * 240 / n, 0] = open_n
                    n_kdata[j + i * 240 / n, 1] = high_n
                    n_kdata[j + i * 240 / n, 2] = low_n
                    n_kdata[j + i * 240 / n, 3] = close_n
                    n_kdata[j + i * 240 / n, 4] = volume_n
    return n_kdata

def get_all_factors(open, high, low, close, volume):
    '''get all factors when n minutes as the period'''
    SMA = talib.MA(close, 30, matype=0)
    EMA = talib.MA(close, 30, matype=1)
    WMA = talib.MA(close, 30, matype=2)
    DEMA = talib.MA(close, 30, matype=3)
    TEMA = talib.MA(close, 30, matype=4)

    MA5 = talib.MA(close, 5, matype=0)
    MA10 = talib.MA(close, 10, matype=0)
    MA20 = talib.MA(close, 20, matype=0)
    MA60 = talib.MA(close, 60, matype=0)
    MA120 = talib.MA(close, 120, matype=0)
    # ======================================== MACD =============================================
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # ========================================= RSI =============================================
    rsi = talib.RSI(close, timeperiod=6)
    # ========================================= KDJ =============================================
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=9,
                               slowk_period=3,
                               slowd_period=3,
                               slowk_matype=0,
                               slowd_matype=0)
    # ========================================== BOLLING ========================================
    upper, middle, lower = talib.BBANDS(close,
                                        timeperiod=26,
                                        # number of non-biased standard deviations from the mean
                                        nbdevup=2,
                                        nbdevdn=2,
                                        # Moving average dataType: simple moving average here
                                        matype=0)
    # =========================================== SAR ============================================
    sar = talib.SAR(high, low, acceleration=0.05, maximum=0.2)

    mat = open                              #0
    mat = np.column_stack((mat, high))      #1
    mat = np.column_stack((mat, low))       #2
    mat = np.column_stack((mat, close))     #3
    mat = np.column_stack((mat, volume))    #4

    mat = np.column_stack((mat, upper))     #5
    mat = np.column_stack((mat, middle))    #6
    mat = np.column_stack((mat, lower))

    mat = np.column_stack((mat, slowk))
    mat = np.column_stack((mat, slowd))

    mat = np.column_stack((mat, macd))
    mat = np.column_stack((mat, macdhist))
    mat = np.column_stack((mat, macdsignal))

    mat = np.column_stack((mat, SMA))
    mat = np.column_stack((mat, EMA))
    mat = np.column_stack((mat, WMA))
    mat = np.column_stack((mat, DEMA))
    mat = np.column_stack((mat, TEMA))

    mat = np.column_stack((mat, rsi))

    mat = np.column_stack((mat, sar))

    mat = np.column_stack((mat, MA5))
    mat = np.column_stack((mat, MA10))
    mat = np.column_stack((mat, MA20))
    mat = np.column_stack((mat, MA60))
    mat = np.column_stack((mat, MA120))
    # np.savetxt('1316.csv', mat, delimiter = ',')
    # 25
    return mat

def get_additional_factors(open, high, low, close, volume):

        # Overlap Studies Functions
        mat = get_all_factors(open, high, low, close, volume)

        mat = np.column_stack((mat,talib.HT_TRENDLINE(close))) ## close
        mat = np.column_stack((mat,talib.KAMA(close,timeperiod=30))) ##close


        #Momentum Indicator Functions
        mat = np.column_stack((mat,talib.ADX(high,low,close,timeperiod=14)))
        mat = np.column_stack((mat,talib.ADXR(high,low,close,timeperiod=14)))
        mat = np.column_stack((mat,talib.APO(close,fastperiod=12,slowperiod=26,matype=0)))
        mat = np.column_stack((mat,talib.AROONOSC(high,low,timeperiod=14)))
        mat = np.column_stack((mat,talib.BOP(open,high,low,close)))
        mat = np.column_stack((mat,talib.MOM(close,timeperiod=10)))

        #Volume Indicator Functions
        mat = np.column_stack((mat, talib.AD(high,low,close,volume)))
        mat = np.column_stack((mat, talib.ADOSC(high,low,close,volume,fastperiod=3,slowperiod=10)))
        mat = np.column_stack((mat, talib.OBV(close,volume)))

        #Volatility Indicator Functions
        mat = np.column_stack((mat, talib.NATR(high,low,close,timeperiod=14)))
        mat = np.column_stack((mat, talib.TRANGE(high,low,close)))

        #Price Transform Functions
        mat = np.column_stack((mat, talib.AVGPRICE(open,high,low,close)))
        mat = np.column_stack((mat, talib.MEDPRICE(high,low)))
        mat = np.column_stack((mat, talib.TYPPRICE(high,low,close)))
        mat = np.column_stack((mat, talib.WCLPRICE(high,low,close)))

        #Cycle Indicator Functions
        mat = np.column_stack((mat, talib.HT_DCPERIOD(close)))
        mat = np.column_stack((mat, talib.HT_DCPHASE(close)))
        mat = np.column_stack((mat, talib.HT_TRENDMODE(close)))

        # 20

        return mat

def get_extend_factors(n, k_data):
    '''extend the n_minutes_period all_factors to 1_minute datas' shape '''
    n_kdata = get_N_KData(n, k_data)
    factors = get_all_factors(n_kdata[:, 0], n_kdata[:, 1], n_kdata[:, 2], n_kdata[:, 3], n_kdata[:, 4])
    extend_data = extend_mat(k_data, factors, n)
    return extend_data


def get_all_periods_factors(periods, k_data):
    '''get the all required periods' factors'''
    for i in range(len(periods) + 1):
        if i == 0:
            factors = get_all_factors(k_data[:, 0], k_data[:, 1], k_data[:, 2], k_data[:, 3], k_data[:, 4])
            factors_mat = np.zeros((factors.shape[1], len(periods) + 1, factors.shape[0]))
            factors_mat[:, i, :] = factors.T
        else:
            factors = get_extend_factors(periods[i - 1], k_data)
            factors_mat[:, i, :] = factors.T

    # data_cut
    #factors_mat = factors_mat[:, :, 119 * np.max(periods) + 119 * np.max(periods) / 240:-4]             ############## cut filtered_tail number
    factors_mat = factors_mat[:, :, 119 * np.max(periods) + 119 * np.max(periods) / 240:]
   # print 'raw_data cut_head_num %i,cut_tail_num %i' %(119 * np.max(periods) + 119 * np.max(periods) / 240,-4)
    return factors_mat

def get_data(index_data_path):
    '''the main function to execut all customed function to get the train_data and train_label
    out_put:
    data_sets: shape(25,6,148211), axis=0 means 25 indexes, axis=1 means 6 kinds of time_periods statistics
              axis=2 means time axis, the meaniing of every index presented in axis=0 explained by readme.txt_file
    label_sets: the filtered close_price of HUSHEN300 index as label_set
    '''

    with open(index_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header = next(reader)
        index = [(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])) for row in reader]
        data = np.array(index)
    k_data = data

    periods = [5, 15, 30, 60, 240]
    data_sets = get_all_periods_factors(periods, k_data)
    print 'get_data()/data_sets.shape', data_sets.shape

    return data_sets

# ============================================ get label ================================================================
def genData(filtered_path):
    time_serias = []
    filtered_close_index = []
    with open(filtered_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header = next(reader)
        for i, row in enumerate(reader):
            time_serias.append(i)
            filtered_close_index.append(float(row[1]))
    return time_serias, filtered_close_index


def knee_point(time_serias, filtered_close):
    knee_t = []
    knee_close = []
    for i, v in enumerate(time_serias[1:-1]):
        j = i + 1  # 从第1 个数据开始考虑拐点
        if i == 0 or i == len(time_serias[1:-1]) - 1:
            knee_t.append(v)
            knee_close.append(filtered_close[j])
        elif (filtered_close[j] - filtered_close[j - 1]) * (filtered_close[j + 1] - filtered_close[j]) < 0:
            knee_t.append(v)
            knee_close.append(filtered_close[j])
        else:
            pass

    return knee_t, knee_close


def get_stdr(knee_close):
    delta = np.array(knee_close[1:]) - np.array(knee_close[:-1])
    rate = delta / np.array(knee_close[:-1])
    mean = gp.mean(rate)
    stdr = mean * 50
    return stdr


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


def get_label_list(b_t, b_close, filtered_close):
    x_label_list = []
    y_label_list = []
    for i, v in enumerate(b_t):
        if i == 0:
            continue
        if b_close[i] - b_close[i - 1] > 0:
            flag = 1
        else:
            flag = -1
        n = v - b_t[i - 1]
        for j in range(n):
            x_label_list.append(flag * (n - j))
            y_label = 100 * (b_close[i] - filtered_close[v - n + j]) / filtered_close[v - n + j]
            y_label_list.append(y_label)
    return x_label_list, y_label_list


def get_index_close(index_data_path):
    with open(index_data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header = next(reader)
        index = [float(row[1]) for row in reader]
        index_close = np.array(index)
    return index_close


def get_data_sets(index_data_path, filtered_path, stdr = 0.001):
    t, close = genData(filtered_path)                          ################################## filtered data already cut 5/4 data at head/tail
    print 'filtered data length: %i ' %len(close)
    close = close[119*241-5:]                                  ##################################  head_cut number
    t = range(len(close))
    knee_t, knee_close = knee_point(t, close)  # waste time
    b_t, b_close = break_point(stdr, knee_t, knee_close, t, close)  # waste time
    print 'last break point %i ' %b_t[-1]
    x_label_list, y_label_list = get_label_list(b_t, b_close, close)
    # x_label_list = x_label_list[(119 * 241 - 10):]
    # y_label_list = y_label_list[(119 * 241 - 10):]
    label = (x_label_list, y_label_list)
    data = get_data(index_data_path)
    #data = data[:, :, :-(len(close) - b_t[-1])]                  ################################## tail_cut number
    data = data[:, :, :]
    res_data_num = (len(close) - b_t[-1])
    print 'data.shape', data.shape
    return data,label

def get_latest_info(index_data_path, filtered_path, stdr = 0.001):
    t, close = genData(filtered_path)
    latest_num = len(close)+9-176894 # 176894 is the number of three year
    knee_t, knee_close = knee_point(t, close)  # waste time
    b_t, b_close = break_point(stdr, knee_t, knee_close, t, close)  # waste time
    cut_tail_num = len(close) - (b_t[-1])+ 4                     #############################  get tail_cut number
    return  latest_num, cut_tail_num

if __name__ == '__main__':

    ### test
    filtered_path = "/home/zqfrgzn04/datasource/three_and_half_filtered_10.csv"
    index_data_path = "/home/zqfrgzn04/datasource/three_and_half_min.csv"
    t, close = genData(filtered_path)

    close = close[119*241-5:]
    print len(close)
    t = range(len(close))