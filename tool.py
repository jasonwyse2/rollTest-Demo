#coding:utf-8
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import pymysql
import scipy.signal as signal
from numpy import random
from pandas import DataFrame
import datetime
import time
def roll_test():
    print('roll_test')


from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

myfont = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=10)
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def save_fig(close_attach_y, close, y, fig_path):
    # with PdfPages(fig_path) as pdf:
    # pp = PdfPages(fig_path)
    # with PdfPages(fig_path) as pdf:
    # As many times as you like, create a figure fig and save it:

    fig = plt.figure()
    # pdf.savefig(fig)
    point_size = 20
    plt.plot(range(close.shape[0]), close_attach_y)
    plt.plot(range(close.shape[0]), close)
    plt.scatter(np.where(y == 0)[0], close_attach_y[np.where(y == 0)[0]], marker='o', c='red', label=u'急涨', s=point_size, )
    plt.scatter(np.where(y == 1)[0], close_attach_y[np.where(y == 1)[0]], marker='o', c='green', label=u'急跌', s=point_size)
    plt.scatter(np.where(y == 2)[0], close_attach_y[np.where(y == 2)[0]], marker='o', c='violet', label=u'缓涨', s=point_size)
    plt.scatter(np.where(y == 3)[0], close_attach_y[np.where(y == 3)[0]], marker='o', c='lightgreen', label=u'缓跌', s=point_size)
    plt.legend(prop=myfont)
    plt.savefig(fig_path, dpi=fig.dpi)

def show_confuse_matrix(y_true,y_predict):
    # confusion_matrix(y_true, y_pred), be aware y_true and y_pred should have the same type
    confusion_result = confusion_matrix(np.array(y_true), np.array(y_predict),labels=range(4))
    confu_matrix = pd.DataFrame(confusion_result)
    # column is predict
    print('        *** Confusion Matrix ***')
    print('====================================================')
    column_list = []
    for i in range(confu_matrix.shape[1]):
        column_list.append('pred ' + str(i))
    confu_matrix.columns = column_list
    index_list = []
    for i in range(confu_matrix.shape[0]):
        index_list.append('real ' + str(i))
    confu_matrix.index = index_list
    print confu_matrix
    return confu_matrix

def make_directory(dir):
    '''
    :param dir: if dir exist, do nothing, else create a director named 'dir'
    :return:
    '''
    if  not os.path.exists(dir):
        os.makedirs(dir)

def currentTime_toString():
    ISOTIMEFORMAT = '%Y-%m-%d-%H-%M-%S'
    currenttime_str = time.strftime(ISOTIMEFORMAT, time.localtime())
    return currenttime_str

def get_onlyFileName(parameter_dict):
    dataType = parameter_dict['dataType']
    if dataType == 'train_valid':
        middle_str = parameter_dict['train_endTime']
    elif dataType == 'test':
        middle_str = ''
    else:
        raise Exception('illegal dataType:%s' % dataType)
    startTime, endTime, train_lookBack, valid_lookBack = get_start_end_lookback(parameter_dict)
    start_str = startTime if not startTime == '' else 'StartNotGiven'
    train_lookBack_str = str(train_lookBack) if not str(train_lookBack) == '0' else ''
    valid_lookBack_str = str(valid_lookBack)
    code_wind = parameter_dict['code_wind']

    file_name = code_wind + '-' + start_str + '-' + middle_str + '-' + endTime + '-' \
                + train_lookBack_str + '-' + valid_lookBack_str + '-' + dataType
    return file_name

def get_underlyingTime_directory(parameter_dict):
    project_directory = parameter_dict['project_directory']
    code_wind = parameter_dict['code_wind']
    if not parameter_dict.has_key('currenttime_str'):
        currenttime_str = currentTime_toString()
        parameter_dict['currenttime_str'] = currenttime_str
    underlyingTime_directory = project_directory+ code_wind + '_' + parameter_dict['currenttime_str'] + '/'
    return underlyingTime_directory

def login_MySQL(db_parameter={}):
    db_host = db_parameter['host']
    db_port = db_parameter['port']
    db_user = db_parameter['user']
    db_pass = db_parameter['passwd']
    db_name = db_parameter['db']
    conn = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name)
    return  conn

def currentDay_forward_delta(currentTime, deltaTime):

    curr = datetime.datetime.strptime(currentTime, '%Y%m%d')
    delta = datetime.timedelta(deltaTime)
    cutTime = (curr + delta).strftime('%Y%m%d')
    return cutTime

def currentDay_backward_delta(currentTime, deltaTime):
    curr = datetime.datetime.strptime(currentTime, '%Y%m%d')
    delta = datetime.timedelta(deltaTime)
    cutTime = (curr + (-1)* delta).strftime('%Y%m%d')
    return cutTime




def get_codeID_by_codeWind(parameter_dict,code_wind, type):
    conn = login_MySQL(parameter_dict)
    sql = r'select code_id from stock where code_wind= %s and type=%s'%(code_wind, type)
    df = pd.read_sql(sql, conn)
    code_id = df.ix[0,0]
    return code_id

def get_minute_data(parameter_dict, minuteData_usedFor='train'):
    pass

def get_tradeDays_between_start_end(dateStart,dateEnd):
    pass
def get_start_end_lookback(parameter_dict):
    train_lookBack = 0
    valid_lookBack = 0
    if parameter_dict['dataType'] == 'test':
        startTime = parameter_dict['test_startTime']
        valid_lookBack = parameter_dict['test_lookBack_from_endTime']
        endTime = parameter_dict['test_endTime']
        if not startTime=='':
            dataType = parameter_dict['dataType']
            parameter_dict['dataType'] = 'test'
            train_df = get_DF_between_start_end(parameter_dict['test_startTime'],parameter_dict['test_endTime'],parameter_dict)
            valid_lookBack = train_df.shape[0]
            #### restore the original value of 'parameter_dict['dataType']'
            parameter_dict['dataType'] = dataType
        else:
            valid_lookBack = parameter_dict['valid_lookBack_from_endTime']

    elif parameter_dict['dataType'] == 'valid':
        startTime = parameter_dict['valid_startTime']
        valid_lookBack = parameter_dict['valid_lookBack_from_endTime']
        endTime = parameter_dict['valid_endTime']
    elif parameter_dict['dataType'] == 'train':
        startTime = parameter_dict['train_startTime']
        train_lookBack = parameter_dict['train_lookBack_from_endTime']
        endTime = parameter_dict['train_endTime']
    elif parameter_dict['dataType'] == 'train_valid':
        startTime = parameter_dict['train_startTime']
        endTime = parameter_dict['valid_endTime']
        if not startTime=='':
            dataType = parameter_dict['dataType']
            parameter_dict['dataType'] = 'train'
            train_df = get_DF_between_start_end(parameter_dict['train_startTime'],parameter_dict['train_endTime'],parameter_dict)
            train_lookBack = train_df.shape[0]
            parameter_dict['dataType'] = 'valid'
            valid_df = get_DF_between_start_end(parameter_dict['valid_startTime'], parameter_dict['valid_endTime'], parameter_dict)
            valid_lookBack = valid_df.shape[0]
            #### restore the original value of 'parameter_dict['dataType']'
            parameter_dict['dataType'] = dataType
        else:
            train_lookBack = parameter_dict['train_lookBack_from_endTime']
            valid_lookBack = parameter_dict['valid_lookBack_from_endTime']

    return startTime,endTime,train_lookBack, valid_lookBack


def get_DF_between_start_end(startTime,endTime,parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s ' % field + 'from %s ' % table + \
          'where code_id = %s  and date>= %s and date<%s ' % (code_id, startTime, endTime)
    df_start_end = pd.read_sql(sql, conn)
    return df_start_end

def get_DF_NTradeDays_before_startTime(startTime,NTradeDays_before_startTime,parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s' % field + 'from %s ' % table + 'where code_id = %s and date < %s ' % (code_id, startTime) \
          + 'order by date limit %s ' % NTradeDays_before_startTime
    df_indicatorDays = pd.read_sql(sql, conn)
    return df_indicatorDays

def get_DF_NTradeDays_after_endTime(endTime,NTradeDays_after_endTime,parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s' % field + 'from %s ' % table + 'where code_id = %s and date >= %s ' % (code_id, endTime) \
          + 'order by date limit %s ' % NTradeDays_after_endTime
    df_indicatorDays = pd.read_sql(sql, conn)
    return df_indicatorDays


def get_daily_data(parameter_dict):
    '''
    Besides the data pointed by the startTime and endTime. It also contains top NTradeDays for indicatorCalculation,
    and contains extra days for tagging data. It would raise Exception when there is no enough days after endTime.
    :param parameter_dict:
    :return:
    '''
    dataType = parameter_dict['dataType']
    startTime, endTime, train_lookBack,valid_lookBack  = get_start_end_lookback(parameter_dict)

    if dataType=='train_valid':
        parameter_dict['valid_lookBack_from_endTime'] = valid_lookBack
        parameter_dict['train_lookBack_from_endTime'] = train_lookBack
        lookBack_from_endTime = train_lookBack + valid_lookBack
    elif dataType=='test': #### dataType=='test'
        parameter_dict['test_lookBack_from_endTime'] = valid_lookBack
        lookBack_from_endTime = valid_lookBack
    else:
        raise Exception('illegal dataType:%s, not support now'%dataType)
    #######  get stock's unique 'code_id' (primary key) with 'code_wind' and 'stock_type' in table 'stock'
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict,code_wind,stock_type)

    table,field = parameter_dict['db_table'],parameter_dict['db_table_field']
    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    conn = login_MySQL(parameter_dict)
    Ndays_backward = int(NTradeDays_for_indicatorCalculation) + int(lookBack_from_endTime)
    sql = 'select %s' % field + 'from %s ' % table + 'where code_id = %s and date < %s '%(code_id,endTime)\
          + 'order by date desc limit %s '%Ndays_backward
    df_data_descendByDate = pd.read_sql(sql, conn)
    df_data_ascendByDate = df_data_descendByDate.reindex(range(len(df_data_descendByDate.index)-1,-1,-1))
    df_data_ascendByDate.index = range(len(df_data_descendByDate.index))
    extraTradeDays_afterEndTime = parameter_dict['extraTradeDays_afterEndTime_for_filter']
    df_extraTradeDays = get_DF_NTradeDays_after_endTime(endTime, extraTradeDays_afterEndTime,
                                                        parameter_dict)
    real_extraTradeDays_readFromDB = df_extraTradeDays.shape[0]
    if real_extraTradeDays_readFromDB < extraTradeDays_afterEndTime:
        raise Exception('there is no enough extra days(%s) after:%s' % (extraTradeDays_afterEndTime, endTime))
    df_extraTradeDays.index = range(len(df_data_descendByDate.index),len(df_data_descendByDate.index)+len(df_extraTradeDays))
    df = pd.concat([df_data_ascendByDate,df_extraTradeDays])
    return df

def get_fileName_by_startEndLookback(parameter_dict, dataType = 'train_valid'):
    if dataType == 'train_valid':
        middle_str = parameter_dict['train_endTime']
    else:
        middle_str = ''
    dailyData_usedFor = parameter_dict['dataType']
    startTime, endTime, train_lookBack,valid_lookBack = get_start_end_lookback(parameter_dict)
    start_str = startTime if not startTime=='' else 'StartNotGiven'
    lookBack_from_endTime_str = str(train_lookBack) if not '' else 'LookBackNotGiven'
    valid_lookBack_str = str(valid_lookBack)
    code_wind = parameter_dict['code_wind']
    simulativeCloseSeries_directory = parameter_dict['simulativeCloseSeries_directory']
    postfix = parameter_dict['dataFile_postfix']
    make_directory(simulativeCloseSeries_directory)
    name = simulativeCloseSeries_directory+code_wind+'-'+start_str+'-'+middle_str+'-'+endTime+'-'\
           +lookBack_from_endTime_str+'-'+valid_lookBack_str+'-'+dailyData_usedFor
    fileName = name+postfix
    return fileName

def simulative_close_generator(date, close_price, series_num, filter_window_size = 5, kaiser_beta = 2, noise_window_size = 5):
    # fake high frequency single generate
    window = signal.kaiser(filter_window_size, beta=kaiser_beta)
    close_filtered = signal.convolve(close_price, window, mode='same') / sum(window)
    high_frequence = close_price - close_filtered
    count = 0
    simulativeNoiseSerise_list = []
    while count < series_num:
        count += 1
        simulativeNoiseSerise = []
        random_location = random.randint(0, noise_window_size)
        close_len = list(high_frequence)
        for i in range(len(close_len) - (noise_window_size-1)):
            simulativeNoiseSerise.append(close_len[i + random_location])

        for i in range(len(close_len) - (noise_window_size-1), len(close_len)):
            random_location2 = random.randint(0, len(close_len) - i)
            simulativeNoiseSerise.append(close_len[i + random_location2])
        simulativeNoiseSerise_list.append(simulativeNoiseSerise)
    # fake data generate
    fake_data = []
    for i in range(series_num):
        add = [filtered_close + noise for filtered_close, noise in zip(close_filtered, simulativeNoiseSerise_list[i] )]
        fake_data.append(add)
    fake_data = np.array(fake_data).T
    # index generate
    index_date = list(date)
    index_columns = []
    for i in range(series_num):
        index_columns.append('series ' + str(i))
    fake_data = DataFrame(fake_data, index=index_date, columns=index_columns)
    return fake_data

if __name__ == '__main__':
    # conn = login_MySQL()
    # sql = 'select * from stock_date where code_id = 1'
    # df = pd.read_sql(sql, conn, )
    # result = pd.DataFrame(df)
    # print result
    #print currentTime_toString()
    pass
