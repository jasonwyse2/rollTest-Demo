import numpy as np
import talib
# import util.get_3d_data_set as get3d
import datetime
# from scipy.stats import norm
import scipy.signal as signal
import data.datalib.util.get_3d_data_set as get3d
from data.ProbabilityIndicator import get_probability_indicator
indicator_comb_list = ['comb_return_1', 'comb_return_2', 'comb_return_3']

def get_indicator_combination(indicatorCombinationName):
    if indicatorCombinationName=='comb_return_1':
        return get_indicator_comb1
        #mat = get_indicator_comb1()
    elif indicatorCombinationName=='comb_return_2':
        return get_indicator_comb2
    elif indicatorCombinationName=='comb_return_3':
        return get_indicator_comb3


def get_indicator_comb1(close_return):
    indicator_list = []
    BBANDS_parameter_list = [(26,3,3),(13,2,2),(5,2,2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_return, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    WMA = talib.MA(close_return, 30, matype=2)
    indicator_list.append(WMA)
    TEMA = talib.MA(close_return, 30, matype=4)
    indicator_list.append(TEMA)
    rsi = talib.RSI(close_return, timeperiod=6)
    indicator_list.append(rsi)
    macd_macdsignal_macdhist = talib.MACD(close_return, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_macdsignal_macdhist:
        indicator_list.append(v)
    elas_p = get_probability_indicator(close_return, lmd_1=0.475, lmd_2=0.4)
    for v in elas_p:
        indicator_list.append(v)
    mat = close_return
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat
def get_indicator_comb2(close_return):
    indicator_list = []
    BBANDS_parameter_list = [(26,3,3),(13,2,2),(5,2,2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_return, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    WMA = talib.MA(close_return, 30, matype=2)
    indicator_list.append(WMA)
    TEMA = talib.MA(close_return, 30, matype=4)
    indicator_list.append(TEMA)
    rsi = talib.RSI(close_return, timeperiod=6)
    indicator_list.append(rsi)
    macd_macdsignal_macdhist = talib.MACD(close_return, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_macdsignal_macdhist:
        indicator_list.append(v)
    elas_p = get_probability_indicator(close_return, lmd_1=0.475, lmd_2=0.4)
    for v in elas_p:
        indicator_list.append(v)
    mat = close_return
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat

def get_indicator_comb3(close_return):
    indicator_list = []
    BBANDS_parameter_list = [(26,3,3),(13,2,2),(5,2,2)]
    for i in range(len(BBANDS_parameter_list)):
        upper_middle_lower = talib.BBANDS(close_return, timeperiod=BBANDS_parameter_list[i][0], nbdevup=BBANDS_parameter_list[i][1],
                                          nbdevdn=BBANDS_parameter_list[i][2], matype=0)
        for v in upper_middle_lower:
            indicator_list.append(v)
    WMA = talib.MA(close_return, 30, matype=2)
    indicator_list.append(WMA)
    TEMA = talib.MA(close_return, 30, matype=4)
    indicator_list.append(TEMA)
    rsi = talib.RSI(close_return, timeperiod=6)
    indicator_list.append(rsi)
    macd_macdsignal_macdhist = talib.MACD(close_return, fastperiod=12, slowperiod=26, signalperiod=9)
    for v in macd_macdsignal_macdhist:
        indicator_list.append(v)
    elas_p = get_probability_indicator(close_return, lmd_1=0.475, lmd_2=0.4)
    for v in elas_p:
        indicator_list.append(v)
    mat = close_return
    for v in indicator_list:
        mat = np.column_stack((mat, v))
    return mat
if __name__ == '__main__':
    close_return = np.array([1.0]*101)
    mat = get_indicator_comb1(close_return)
    result = talib.BBANDS(close_return, timeperiod=26, nbdevup=3, nbdevdn=3, matype=0)
    res = []
    for item in result:
        res.append(item)
    print(res)