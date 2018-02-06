import numpy as np
import talib
# import util.get_3d_data_set as get3d
import datetime
# from scipy.stats import norm
import scipy.signal as signal
import data.datalib.util.get_3d_data_set as get3d
from data.ProbabilityIndicator import get_probability_indicator
day_comb_list = ['dayComb1', 'dayComb2','dayComb3']
minute_comb_list = ['minuteComb1']
def get_indicator_handle(indicatorCombinationName):
    if indicatorCombinationName=='dayComb1':
        return get_indicator_day_comb1
    elif indicatorCombinationName=='dayComb2':
        return get_indicator_comb2
    elif indicatorCombinationName=='minuteComb1':
        return get_indicator_minute_comb1

def get_indicator_day_comb1(raw_data_df,indicator_combination=''):
    # if indicator_combination not in day_comb_list:
    #     raise Exception('"indicator_combination" does not match the variable "dayOrMinute"')
    close = np.array(raw_data_df.iloc[:,0])
    close_pct = np.diff(close) / close[:-1]
    indicator_list = []
    BBANDS_parameter_list = [(26,3,3),(13,2,2),(5,2,2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_pct, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    WMA = talib.MA(close_pct, 30, matype=2)
    indicator_list.append(WMA)
    TEMA = talib.MA(close_pct, 30, matype=4)
    indicator_list.append(TEMA)
    rsi = talib.RSI(close_pct, timeperiod=6)
    indicator_list.append(rsi)
    macd_macdsignal_macdhist = talib.MACD(close_pct, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_macdsignal_macdhist:
        indicator_list.append(v)
    elas_p = get_probability_indicator(close_pct, lmd_1=0.475, lmd_2=0.4)
    for v in elas_p:
        indicator_list.append(v)
    ########### merge all indicator into one ################
    mat = close_pct
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat
def get_indicator_comb2(raw_data_df,indicator_combination):
    open, high, low, close = raw_data_df[0], raw_data_df[1], raw_data_df[2], raw_data_df[3]
    close_pct = np.diff(close) / close[:-1]

    indicator_list = []
    BBANDS_parameter_list = [(26,3,3),(13,2,2),(5,2,2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_pct, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    WMA = talib.MA(close_pct, 30, matype=2)
    indicator_list.append(WMA)
    TEMA = talib.MA(close_pct, 30, matype=4)
    indicator_list.append(TEMA)
    rsi = talib.RSI(close_pct, timeperiod=6)
    indicator_list.append(rsi)
    macd_macdsignal_macdhist = talib.MACD(close_pct, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_macdsignal_macdhist:
        indicator_list.append(v)
    elas_p = get_probability_indicator(close_pct, lmd_1=0.475, lmd_2=0.4)
    for v in elas_p:
        indicator_list.append(v)
    mat = close_pct
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat

def get_indicator_minute_comb1(raw_data_df,indicator_combination=''):
    '''get all factors when n minutes as the period'''
    # if indicator_combination not in minute_comb_list:
    #     raise Exception('"indicator_combination" does not match the variable "dayOrMinute"')
    open, high = np.array(raw_data_df['open']), np.array(raw_data_df['high'])
    low, close = np.array(raw_data_df['low']), np.array(raw_data_df['close'])

    open_pct = np.diff(open) / open[:-1]
    high_pct = np.diff(high) / high[:-1]
    low_pct = np.diff(low) / low[:-1]
    close_pct = np.diff(close) / close[:-1]

    indicator_list = []
    matype_list = [0,1,2,3,4]
    timeperiod_list = [5,10,20,60,120]
    for i in range(len(matype_list)):
        result = talib.MA(close_pct, 30, matype=matype_list[i])
        indicator_list.append(result)
    for i in range(len(timeperiod_list)):
        result = talib.MA(close_pct, timeperiod_list[i], matype=0)
        indicator_list.append(result)
    # ======================================== MACD =============================================
    macd_list = talib.MACD(close_pct, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_list:
        indicator_list.append(v)
    # ========================================= RSI =============================================
    rsi = talib.RSI(close_pct, timeperiod=6)
    indicator_list.append(rsi)
    # ========================================= KDJ =============================================
    slowk_slowd = talib.STOCH(high_pct, low_pct, close_pct, fastk_period=9,
                               slowk_period=3, slowd_period=3, slowk_matype=0, slowd_matype=0)
    for v in slowk_slowd:
        indicator_list.append(v)
    # ========================================== BOLLING ========================================
    BBANDS_parameter_list = [(26, 2, 2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_pct, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    # =========================================== SAR ============================================
    sar = talib.SAR(high_pct, low_pct, acceleration=0.05, maximum=0.2)
    indicator_list.append(sar)
    mat = open_pct
    price_list = [high_pct,low_pct,close_pct]
    for i in range(1, len(price_list)):
        np.column_stack((mat, price_list[i]))
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat
        # SMA = talib.MA(close, 30, matype=0)
    # EMA = talib.MA(close, 30, matype=1)
    # WMA = talib.MA(close, 30, matype=2)
    # DEMA = talib.MA(close, 30, matype=3)
    # TEMA = talib.MA(close, 30, matype=4)

    # MA5 = talib.MA(close, 5, matype=0)
    # MA10 = talib.MA(close, 10, matype=0)
    # MA20 = talib.MA(close, 20, matype=0)
    # MA60 = talib.MA(close, 60, matype=0)
    # MA120 = talib.MA(close, 120, matype=0)
    # ======================================== MACD =============================================
    # macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # ========================================= RSI =============================================
    # rsi = talib.RSI(close, timeperiod=6)
    # # ========================================= KDJ =============================================
    # slowk, slowd = talib.STOCH(high, low, close, fastk_period=9,
    #                            slowk_period=3,
    #                            slowd_period=3,
    #                            slowk_matype=0,
    #                            slowd_matype=0)
    # # ========================================== BOLLING ========================================
    # upper, middle, lower = talib.BBANDS(close,
    #                                     timeperiod=26,
    #                                     # number of non-biased standard deviations from the mean
    #                                     nbdevup=2,
    #                                     nbdevdn=2,
    #                                     # Moving average type: simple moving average here
    #                                     matype=0)
    # # =========================================== SAR ============================================
    # sar = talib.SAR(high, low, acceleration=0.05, maximum=0.2)

    # mat = open                              #0
    # mat = np.column_stack((mat, high))      #1
    # mat = np.column_stack((mat, low))       #2
    # mat = np.column_stack((mat, close))     #3
    # # mat = np.column_stack((mat, volume))    #4
    #
    # mat = np.column_stack((mat, upper))     #5
    # mat = np.column_stack((mat, middle))    #6
    # mat = np.column_stack((mat, lower))
    #
    # mat = np.column_stack((mat, slowk))
    # mat = np.column_stack((mat, slowd))
    #
    # mat = np.column_stack((mat, macd))
    # mat = np.column_stack((mat, macdhist))
    # mat = np.column_stack((mat, macdsignal))
    #
    # mat = np.column_stack((mat, SMA))
    # mat = np.column_stack((mat, EMA))
    # mat = np.column_stack((mat, WMA))
    # mat = np.column_stack((mat, DEMA))
    # mat = np.column_stack((mat, TEMA))
    #
    # mat = np.column_stack((mat, rsi))
    #
    # mat = np.column_stack((mat, sar))
    #
    # mat = np.column_stack((mat, MA5))
    # mat = np.column_stack((mat, MA10))
    # mat = np.column_stack((mat, MA20))
    # mat = np.column_stack((mat, MA60))
    # mat = np.column_stack((mat, MA120))
    # np.savetxt('1316.csv', mat, delimiter = ',')
    # 25
    return mat


if __name__ == '__main__':
    close_pct = np.array([1.0] * 101)
    mat = get_indicator_day_comb1(close_pct)
    result = talib.BBANDS(close_pct, timeperiod=26, nbdevup=3, nbdevdn=3, matype=0)
    res = []
    for item in result:
        res.append(item)
    print(res)