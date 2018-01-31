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

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

myfont = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=20)
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def save_figure_pdf_truePredictTogether(filtered_close, close, date, y_list, fig_path, code_wind,
                                label = ['sharp up','sharp down','gentle up','gentle down'],
                                color = ['red','green','violet','lightgreen'],
                                interval=5):

    filtered_close, close, date = np.array(filtered_close), np.array(close), np.array(date)
    point_size = 80
    true_labels_title = code_wind + '-True labels'+' (Time: %s-%s)'%( str(date[0][0]),str(date[-1][0]))
    predict_labels_title = code_wind + '-Predict labels'+' (Time: %s-%s)'%(str(date[0][0]), str(date[-1][0]))
    title_list = [true_labels_title, predict_labels_title]
    with PdfPages(fig_path) as pdf:
        plt.figure(figsize=(40, 25))#
        figure_position = [211,212]
        plt.title(code_wind)
        for i in range(len(y_list)):
            ax = plt.subplot(figure_position[i])
            plt.sca(ax)
            y = y_list[i]
            plt.title(title_list[i],fontsize=20)
            plt.plot(range(close.shape[0]), filtered_close)
            plt.plot(range(close.shape[0]), close)
            plt.scatter(np.where(y == 0)[0], close[np.where(y == 0)[0]], marker='o', c=color[0], label=label[0], s=point_size)
            plt.scatter(np.where(y == 1)[0], close[np.where(y == 1)[0]], marker='o', c=color[1], label=label[1], s=point_size)
            plt.scatter(np.where(y == 2)[0], close[np.where(y == 2)[0]], marker='o', c=color[2], label=label[2], s=point_size)
            plt.scatter(np.where(y == 3)[0], close[np.where(y == 3)[0]], marker='o', c=color[3], label=label[3], s=point_size)

            plt.legend(fontsize=20)
            interval_idx_list = []
            interval_date_list = []
            for i in range(close.shape[0]):
                if i % interval == 0:
                    interval_idx_list.append(i)
                    interval_date_list.append(date[i][0])
            plt.xticks(tuple(interval_idx_list), tuple(interval_date_list))
            plt.xticks(rotation=80, fontsize=15)
            plt.grid()
            #plt.xlabel('time',fontsize=20)

        pdf.savefig()
        plt.close()

def save_figure_pdf(filtered_close, close, date, y, fig_path, label = ['sharp up','sharp down','gentle up','gentle down'],
                    color=['red', 'green', 'violet', 'lightgreen'], interval=5):

    filtered_close, close, date = np.array(filtered_close), np.array(close), np.array(date)
    point_size = 30
    #label1 = ['sharp up','sharp down','gentle up','gentle down']
    with PdfPages(fig_path) as pdf:
        plt.figure(figsize=(30, 15))
        plt.plot(range(close.shape[0]), filtered_close)
        plt.plot(range(close.shape[0]), close)
        plt.scatter(np.where(y == 0)[0], close[np.where(y == 0)[0]], marker='o', c=color[0], label=label[0], s=point_size)
        plt.scatter(np.where(y == 1)[0], close[np.where(y == 1)[0]], marker='o', c=color[1], label=label[1], s=point_size)
        plt.scatter(np.where(y == 2)[0], close[np.where(y == 2)[0]], marker='o', c=color[2], label=label[2], s=point_size)
        plt.scatter(np.where(y == 3)[0], close[np.where(y == 3)[0]], marker='o', c=color[3], label=label[3],
                    s=point_size)
        #plt.legend(prop=myfont)
        plt.legend(fontsize=20)
        interval_idx_list = []
        interval_date_list = []
        for i in range(close.shape[0]):
            if i % interval == 0:
                interval_idx_list.append(i)
                interval_date_list.append(date[i])
        plt.xticks(tuple(interval_idx_list), tuple(interval_date_list))
        plt.xticks(rotation=80, fontsize=13)
        plt.xlabel('time',fontsize=20)
        pdf.savefig()
        plt.close()



def save_figure_pdf_kneeUpDown(filtered_close, close, date, y, fig_path):

    filtered_close, close, date = np.array(filtered_close), np.array(close), np.array(date)
    point_size = 30
    plt.figure(figsize=(20, 10))
    plt.xticks(range(close.shape[0]), date)
    plt.xticks(rotation=60, fontsize =13)
    plt.plot(range(close.shape[0]), filtered_close)
    plt.plot(range(close.shape[0]), close)
    plt.scatter(np.where(y == 0)[0], close[np.where(y == 0)[0]], marker='o', c='red', label=u'谷底', s=point_size)
    plt.scatter(np.where(y == 1)[0], close[np.where(y == 1)[0]], marker='o', c='green', label=u'上涨', s=point_size)
    plt.scatter(np.where(y == 2)[0], close[np.where(y == 2)[0]], marker='o', c='violet', label=u'顶点', s=point_size)
    plt.scatter(np.where(y == 3)[0], close[np.where(y == 3)[0]], marker='o', c='lightgreen', label=u'下跌', s=point_size)
    plt.legend(prop=myfont)
    plt.savefig(fig_path)
    plt.close()
def show_fig(labels,filter_data_for_use,close_for_use,label = ['sharp up','sharp down','gentle up','gentle down']):

    import matplotlib.pyplot as plt
    labels = labels.astype(int)
    filter_data_for_use = np.array(filter_data_for_use)
    point_size = 20
    plt.plot(range(filter_data_for_use.shape[0]),filter_data_for_use)
    plt.plot(range(filter_data_for_use.shape[0]),close_for_use)
    plt.scatter(np.where(labels == 0)[0],close_for_use[np.where(labels==0)[0]],marker='o',c='red',label=label[0],s=point_size)
    plt.scatter(np.where(labels == 1)[0], close_for_use[np.where(labels == 1)[0]], marker='o',c='green',label=label[1],s=point_size)
    plt.scatter(np.where(labels == 2)[0], close_for_use[np.where(labels == 2)[0]], marker='o', c='violet', label=label[2], s=point_size)
    plt.scatter(np.where(labels == 3)[0], close_for_use[np.where(labels == 3)[0]], marker='o', c='lightgreen', label=label[3], s=point_size)
    plt.legend(prop=myfont)
    plt.show()

def confuse_matrix(y_true, y_predict, parameter_dict):
    # confusion_matrix(y_true, y_pred), be aware y_true and y_pred should have the same type
    nb_classes = parameter_dict['nb_classes']
    confusion_result = confusion_matrix(np.array(y_true), np.array(y_predict),labels=range(nb_classes))
    confuse_matrix = pd.DataFrame(confusion_result)
    # column is predict
    # print('        *** Confusion Matrix ***')
    # print('====================================================')
    column_list = []
    for i in range(confuse_matrix.shape[1]):
        column_list.append('pred ' + str(i))
    confuse_matrix.columns = column_list
    index_list = []
    for i in range(confuse_matrix.shape[0]):
        index_list.append('real ' + str(i))
    confuse_matrix.index = index_list

    #print confuse_matrix
    confuse = confuse_matrix.as_matrix()
    pred_acc_per_class = np.diag(confuse).astype(np.float) / np.sum(confuse, axis=0)
    #### remove 'nan' value in 'pred_acc_per_class'
    pred_acc_per_class[np.where(np.isnan(pred_acc_per_class))]=0
    # print('predict_accuracy_per_class:', pred_acc_per_class)

    startTime, endTime, train_lookBack, valid_lookBack = get_start_end_lookback(parameter_dict)
    confuse_matrix.loc['accuracy(%s-%s)'%(startTime,endTime)] = pred_acc_per_class

    # print('Average accuracy:', np.mean(pred_acc_per_class))
    #average_accuracy_allClass = np.mean(pred_acc_per_class[np.where(pred_acc_per_class>0)])
    average_accuracy_allClass = np.sum(np.diag(confuse).astype(np.float))/np.sum(confuse)
    confuse_matrix.loc['average'] = [average_accuracy_allClass,np.nan,np.nan,np.nan]
    return confuse_matrix

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
    elif dataType == 'train' or dataType == 'valid':
        middle_str = ''
    else:
        raise Exception('illegal dataType:%s' % dataType)
    startTime, endTime, train_lookBack, valid_lookBack = get_start_end_lookback(parameter_dict)
    start_str = startTime if not startTime == '' else 'StartNotGiven'
    train_lookBack_str = str(train_lookBack) if not str(train_lookBack) == '0' else ''
    valid_lookBack_str = str(valid_lookBack) if not str(valid_lookBack) == '0' else ''
    code_wind = parameter_dict['code_wind']

    file_name = code_wind + '-' + start_str + '-' + middle_str + '-' + endTime + '-' \
                + train_lookBack_str + '-' + valid_lookBack_str + '-' + dataType
    return file_name

def get_underlyingTime_directory(parameter_dict):
    project_directory = parameter_dict['project_directory']
    code_wind = parameter_dict['code_wind']
    task_description = parameter_dict['task_description']
    if not parameter_dict.has_key('currenttime_str'):
        currenttime_str = currentTime_toString()
        parameter_dict['currenttime_str'] = currenttime_str
    underlyingTime_directory = project_directory+ code_wind + '_' + parameter_dict['currenttime_str'] + '_'+task_description+'/'

    return underlyingTime_directory

def login_MySQL(db_parameter={}):
    db_host = db_parameter['host']
    db_port = db_parameter['port']
    db_user = db_parameter['user']
    db_pass = db_parameter['passwd']
    db_name = db_parameter['db']
    conn = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name)
    return  conn

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
        endTime = parameter_dict['test_endTime']
        if not startTime=='':
            valid_df = get_dataframe_between_start_end(startTime, endTime, parameter_dict)
            valid_lookBack = valid_df.shape[0]
        else:
            valid_lookBack = parameter_dict['test_lookBack_from_endTime']

    elif parameter_dict['dataType'] == 'valid':
        startTime = parameter_dict['valid_startTime']
        endTime = parameter_dict['valid_endTime']
        if not startTime=='':
            valid_df = get_dataframe_between_start_end(startTime, endTime, parameter_dict)
            valid_lookBack = valid_df.shape[0]
        else:
            valid_lookBack = parameter_dict['valid_lookBack_from_endTime']

    elif parameter_dict['dataType'] == 'train':
        startTime = parameter_dict['train_startTime']
        endTime = parameter_dict['train_endTime']
        if not startTime=='':
            train_df = get_dataframe_between_start_end(startTime, endTime, parameter_dict)
            train_lookBack = train_df.shape[0]
        else:
            train_lookBack = parameter_dict['train_lookBack_from_endTime']

    elif parameter_dict['dataType'] == 'train_valid':
        startTime = parameter_dict['train_startTime']
        endTime = parameter_dict['valid_endTime']
        if not startTime=='':
            train_df = get_dataframe_between_start_end(parameter_dict['train_startTime'], parameter_dict['train_endTime'], parameter_dict)
            train_lookBack = train_df.shape[0]
            valid_df = get_dataframe_between_start_end(parameter_dict['valid_startTime'], parameter_dict['valid_endTime'], parameter_dict)
            valid_lookBack = valid_df.shape[0]
        else:
            train_lookBack = parameter_dict['train_lookBack_from_endTime']
            valid_lookBack = parameter_dict['valid_lookBack_from_endTime']
    elif parameter_dict['dataType'] == 'rollTestAll':
        startTime = parameter_dict['test_mostStartTime']
        endTime = parameter_dict['test_mostEndTime']
    else:
        startTime = ''
        endTime = ''
    return startTime,endTime,train_lookBack, valid_lookBack


def get_dataframe_between_start_end(startTime, endTime, parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s ' % field + 'from %s ' % table + \
          'where code_id = %s  and date>= %s and date<%s ' % (code_id, startTime, endTime)
    df_start_end = pd.read_sql(sql, conn)
    return df_start_end

def save_allConfuseMatrix(allConfuseMatrix, parameter_dict, type = 'train'):
    code_wind = parameter_dict['code_wind']
    test_mostStartTime =  parameter_dict['test_mostStartTime']
    test_mostEndTime = parameter_dict['test_mostEndTime']
    underlyingTime_directory = get_underlyingTime_directory(parameter_dict)
    allConfuseMatrix_fileName = underlyingTime_directory + code_wind+'-roll_ConfuseMatrix-'+type+'.csv'
    allConfuseMatrix.to_csv(allConfuseMatrix_fileName, encoding='utf-8')


def get_dataframe_NTradeDays_before_startTime(startTime, NTradeDays_before_startTime, parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s' % field + 'from %s ' % table + 'where code_id = %s and date < %s ' % (code_id, startTime) \
          + 'order by date desc limit %s ' % NTradeDays_before_startTime
    df_data_descendByDate = pd.read_sql(sql, conn)
    df_data_ascendByDate = df_data_descendByDate.reindex(range(len(df_data_descendByDate.index)-1,-1,-1))
    df_data_ascendByDate.index = range(len(df_data_descendByDate.index))
    if df_data_ascendByDate.shape[0] < NTradeDays_before_startTime:
        raise Exception('there is no enough extra days(%s) before:%s' % (NTradeDays_before_startTime, startTime))
    return df_data_ascendByDate

def get_dataframe_NTradeDays_after_endTime(endTime, extraTradeDays_afterEndTime, parameter_dict):
    code_wind, stock_type = parameter_dict['code_wind'], parameter_dict['stock_type']
    code_id = get_codeID_by_codeWind(parameter_dict, code_wind, stock_type)
    conn = login_MySQL(parameter_dict)
    table, field = parameter_dict['db_table'], parameter_dict['db_table_field']
    sql = 'select %s' % field + 'from %s ' % table + 'where code_id = %s and date >= %s ' % (code_id, endTime) \
          + 'order by date limit %s ' % extraTradeDays_afterEndTime
    df_data_ascendByDate = pd.read_sql(sql, conn)

    if df_data_ascendByDate.shape[0] < extraTradeDays_afterEndTime:
        raise Exception('there is no enough extra days(%s) after:%s' % (extraTradeDays_afterEndTime, endTime))
    return df_data_ascendByDate


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
    elif dataType=='valid': #### dataType=='test'
        parameter_dict['valid_lookBack_from_endTime'] = valid_lookBack
    elif dataType=='test':
        parameter_dict['test_lookBack_from_endTime'] = valid_lookBack
    elif dataType=='train':
        parameter_dict['train_lookBack_from_endTime'] = train_lookBack
    else:
        raise Exception('illegal dataType:%s, not support now'%dataType)

    NTradeDays_for_indicatorCalculation = parameter_dict['NTradeDays_for_indicatorCalculation']
    # extraTradeDays_beforeStartTime_for_filter = parameter_dict['extraTradeDays_beforeStartTime_for_filter']
    filter_windowSize= parameter_dict['filter_windowSize']
    extraTradeDays_afterEndTime = int(filter_windowSize)/2  if filter_windowSize%2==1 else (int(filter_windowSize)/2-1)
    parameter_dict['extraTradeDays_afterEndTime'] = extraTradeDays_afterEndTime
    #NTradeDays_before_startTime = NTradeDays_for_indicatorCalculation + extraTradeDays_beforeStartTime_for_filter
    NTradeDays_before_startTime = NTradeDays_for_indicatorCalculation
    df_front_extraTradeDays = get_dataframe_NTradeDays_before_startTime(startTime, NTradeDays_before_startTime, parameter_dict)
    df_between_start_end  =get_dataframe_between_start_end(startTime, endTime, parameter_dict)
    df_end_extraTradeDays = get_dataframe_NTradeDays_after_endTime(endTime, extraTradeDays_afterEndTime,
                                                               parameter_dict)
    df = pd.concat([df_front_extraTradeDays,df_between_start_end, df_end_extraTradeDays])
    df.index = range(len(df.index))
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

def simulative_close_generator(raw_data_df, parameter_dict):
    # fake high frequency single generate
    date = raw_data_df.index
    close_price = raw_data_df.close
    simulativeCloseSeries_num = parameter_dict['simulativeCloseSeries_num']
    addNoise_windowSize = parameter_dict['addNoise_windowSize']
    filter_windowSize = parameter_dict['filter_windowSize']
    kaiser_beta = parameter_dict['kaiser_beta']

    window_array = signal.kaiser(filter_windowSize, beta=kaiser_beta)
    close_filtered = signal.convolve(close_price, window_array, mode='same') / sum(window_array)
    high_frequence = close_price - close_filtered
    count = 0
    simulativeNoiseSerise_list = []
    while count < simulativeCloseSeries_num:
        count += 1
        simulativeNoiseSerise = []
        #random.seed(10)
        random_location = random.randint(0, addNoise_windowSize)
        high_frequence = list(high_frequence)
        for i in range(len(high_frequence) - (addNoise_windowSize-1)):
            simulativeNoiseSerise.append(high_frequence[i + random_location])

        for i in range(len(high_frequence) - (addNoise_windowSize-1), len(high_frequence)):
            random_location2 = random.randint(0, len(high_frequence) - i)
            simulativeNoiseSerise.append(high_frequence[i + random_location2])
        simulativeNoiseSerise_list.append(simulativeNoiseSerise)
    # fake data generate
    #fake_data_list = [list(close_price)]
    fake_data_list = []
    for i in range(simulativeCloseSeries_num):
        merge_filteredClose_noise = [filtered_close + noise for filtered_close, noise in zip(close_filtered, simulativeNoiseSerise_list[i] )]
        fake_data_list.append(merge_filteredClose_noise)
    fake_data = np.array(fake_data_list).T
    # index generate
    index_date = list(date)
    index_columns = []
    for i in range(fake_data.shape[1]):
        index_columns.append('series ' + str(i))
    fake_data_df = DataFrame(fake_data, index=index_date, columns=index_columns)
    return fake_data_df

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
if __name__ == '__main__':
    # conn = login_MySQL()
    # sql = 'select * from stock_date where code_id = 1'
    # df = pd.read_sql(sql, conn, )
    # result = pd.DataFrame(df)
    # print result
    #print currentTime_toString()
    pass
    #plt.figure(1)
    plt.figure(2)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    x = np.linspace(0,3,100)
    for i in xrange(2):
        #plt.figure(1)
        #plt.plot(x,np.exp(i*x/3))
        plt.sca(ax1)
        plt.plot(x,np.sin(i*x))
        plt.sca(ax2)
        plt.plot(x,np.cos(i*x))
    plt.show()