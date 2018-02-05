#coding:utf-8
from __future__ import absolute_import
import numpy as np
import pandas as pd
import tool as tool
import time
from base_class import Parameters
from custom_class import Your_Model, Your_Data
import json
from net import get_net_yield
import matplotlib.pyplot as plt
import matplotlib
#from roll import get_roll_start
import roll
import parameter_config as config
#from save import save_figure,save_file
import save
if __name__ == '__main__':
    start = time.time()
    # parameter settings
    # all parameters must be passing by as a dictionary. like: p = Parameters(a = 1, b = 'parameters')
    ############################ prepare data for model to train ##################################
    stock_type = {'SZ_stock': '1', 'SH_stock': '2', 'SZ_index': '100', 'SH_index': '200'}
    hyper_parameter_dict=\
        {'db_information':{'host':'192.168.1.11', 'port':3306, 'user':'zqfordinary', 'passwd':'Ab123456', 'db':'stock'},

         'underlying': '000905', 'stock_type': stock_type['SH_index'],
        'dayOrMinute': 'alpha',  # ['day', 'minute_no_simulative', 'minute_simulative', 'alpha']
         'indicator_combination': 'day_comb_return_1',  # ['day_comb_return_1, minute_comb_return_1']
         'alpha_csv_path': '/mnt/aidata/生成数据/Alpha扰动300/index_240_test.csv',
         'taskType':'BottomTopUpDown',  #[BottomTopUpDown, SharpGentleUpDown]


        'project_directory':'/mnt/aidata/QuantitativePlatform/a-yaogong/',

        'day_train_startTime':'20060101','day_train_endTime': '20180115',
        'day_valid_startTime': '20180101', 'day_valid_endTime': '20180115',
         'day_test_startTime': '20180101', 'day_test_endTime':'20180115',
        'dalily_change_threshold':0.015,

         'minute_train_startTime':'201308010930', 'minute_train_endTime': '201601060930',
         'minute_valid_startTime':'201601010930', 'minute_valid_endTime' :'201601060930',
         'minute_test_startTime':'201601010930', 'minute_test_endTime': '201601060930',
         'minute_change_threshold': 0.001,

         'simulativeCloseSeries_num': 3,
         'roll_forward': 30 * 3,
         'train_num': 1,
         'rollTest_num': 1
         # 'day_table': 'index_date', 'day_field': 'date, open, high, low, close, pct_chg, volume, amt',
         # 'minute_table': 'index_min', 'minute_field': 'time, open, high, low, close, pct_chg, volume, amt',
         }
    # db_information = {'table': 'index_date','host':'192.168.1.11', 'port':3306, 'user':'zqfordinary', 'passwd':'Ab123456', 'db':'stock'}
    #
    # project_directory = '/mnt/aidata/QuantitativePlatform/a-yaogong/'
    # underlying = '000905'
    # stock_type = stock_type['SH_index']
    # dayOrMinute = 'day' # 'minute'
    # day_train_startTime, day_train_endTime = '20060101' , '20180115'
    # day_valid_startTime, day_valid_endTime = '20180115', '20180129'
    # day_test_startTime, day_test_endTime = '20180115', '20180129'
    # ######################################################
    # minute_train_startTime, minute_train_endTime = '201308010930', '201601060930'
    # minute_valid_startTime, minute_valid_endTime = '201601010930', '201601060930'
    # minute_test_startTime, minute_test_endTime = '201601010930', '201601060930'
    ######################################################


    db_information = hyper_parameter_dict['db_information']
    data_parameter_obj = Parameters(
        host=db_information['host'], port=db_information['port'], user=db_information['user'],
        passwd=db_information['passwd'], db=db_information['db'],
        db_table='', db_field='',

        code_wind=hyper_parameter_dict['underlying'], stock_type=hyper_parameter_dict['stock_type'],
        project_directory=hyper_parameter_dict['project_directory'],
        dayOrMinute =  hyper_parameter_dict['dayOrMinute'],
        alpha_csv_path = hyper_parameter_dict['alpha_csv_path'],
        taskType = hyper_parameter_dict['taskType'],
        indicator_combination = hyper_parameter_dict['indicator_combination'],

        NTradeDays_for_indicatorCalculation=120,
        filter_windowSize=5, kaiser_beta=2, addNoise_windowSize=5,

        nb_classes=4,
        dataFile_directoryName='data/',
        dataFile_postfix='.csv',
        task_description='',

        show_label=False,
        simulativeCloseSeries_num = hyper_parameter_dict['simulativeCloseSeries_num'],
        roll_forward = hyper_parameter_dict['roll_forward'],
        # train_startTime=train_startTime, train_endTime=train_endTime,
        # valid_startTime=valid_startTime, valid_endTime=valid_endTime,
        # test_startTime=test_startTime, test_endTime=test_endTime,
        # train_lookBack_from_endTime=0, valid_lookBack_from_endTime=0, test_lookBack_from_endTime=0,
        # train_valid_type = 'train_valid_separate', #train_valid_together
        # dataType='train',
        # label_SharpGentleUpDown=['sharp up', 'sharp down', 'gentle up', 'gentle down'],
        # color_SharpGentleUpDown = ['red', 'green', 'violet', 'lightgreen'],
        # label_BottomTopUpDown = ['bottom', 'up', 'top', 'down'],
        # color_BottomTopUpDown = ['green', 'violet', 'red', 'lightgreen'],
        # dayOrMinute=hyper_parameter_dict['dayOrMinute'],
        # taskType=hyper_parameter_dict['taskType'],  # label_type = 'BottomTopUpDown' , 'SharpGentleUpDown'
    )
    config.set_additional_parameters(hyper_parameter_dict, data_parameter_obj._args)
    # data_parameter_obj._args['test_mostStartTime'] = data_parameter_obj._args['test_startTime']
    # data_parameter_obj._args['test_mostEndTime'] = data_parameter_obj._args['test_endTime']
    # data_parameter_obj._args['task_description'] = data_parameter_obj._args['task_description']+data_parameter_obj._args['taskType']
    # windowSize = data_parameter_obj._args['filter_windowSize']
    # data_parameter_obj._args['extraTradeDays_afterEndTime'] = int(windowSize/2)  if windowSize%2==1 else int(windowSize/2)-1

    model_parameter_obj = Parameters(
        train_num=hyper_parameter_dict['train_num'], rollTest_num = hyper_parameter_dict['rollTest_num'],
        batch_size=512 * 2, nb_classes=4, nb_input=10, evaluate_batch_size=128, evaluate_verbose=0,
        #filter_row =2 , filter_col =1, filter_num = 16,
        loss='categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_directoryName = 'models/',
        modelFile_postfix='',
    )

    roll_result_dict = roll.get_roll_start(data_parameter_obj, model_parameter_obj)
    save.save_file(data_parameter_obj, roll_result_dict)
    save.save_figure(data_parameter_obj, roll_result_dict)

    elapsed = (time.time() - start)
    print("Time used(minute): ", float(elapsed)/60)