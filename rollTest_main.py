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

         'underlying': '000300', 'stock_type': stock_type['SH_index'],
        'dayOrMinute': 'day',  # ['day', 'minuteNoSimulative', 'minuteSimulative', 'alpha', 'alphaMinute']
         'indicator_combination': 'dayComb1',  # ['dayComb1, minuteComb1']
         'alpha_csv_path': '/mnt/aidata/生成数据/Alpha扰动300/min/index_15_min.csv',
         'taskType':'Volatility',  #[BottomTopUpDown, SharpGentleUpDown, Volatility]
         'volatility_window':5,
        'show_label':False,

         'simulativeCloseSeries_num': 1,
         'train_num': 20,

        'project_directory':'/mnt/aidata/QuantitativePlatform/a-yaogong/',

        'day_train_startTime':'20150101','day_train_endTime': '20170101',
        'day_valid_startTime': '', 'day_valid_endTime': '20180201',
         'day_test_startTime': '', 'day_test_endTime':'',
         'day_roll_forward': 1,  # the minumus unit is one months
         'day_rollTest_num': 1,
         'dalily_change_threshold':0.015,

         'minute_train_startTime':'201501010930', 'minute_train_endTime': '201601010930',
         'minute_valid_startTime':'', 'minute_valid_endTime' :'201601060930',
         'minute_test_startTime':'', 'minute_test_endTime': '',
         'minute_roll_forward': 241,  # one day contains 242 trade minutes
         'minute_rollTest_num': 1,
         'minute_change_threshold': 0.001,
         }

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
        show_label = hyper_parameter_dict['show_label'],
        simulativeCloseSeries_num=hyper_parameter_dict['simulativeCloseSeries_num'],

        NTradeDays_for_indicatorCalculation=120,
        filter_windowSize=5, kaiser_beta=2, addNoise_windowSize=5,

        nb_classes=3,
        dataFile_directoryName='data/',
        dataFile_postfix='.csv',
        task_description='', # it's been set in config_other_parameter(), parameter_config.py

        global_parameter_dict={}

    )
    config.set_additional_parameters(hyper_parameter_dict, data_parameter_obj._args)
    data_parameter_dict = data_parameter_obj._args
    model_parameter_obj = Parameters(
        train_num=hyper_parameter_dict['train_num'], rollTest_num = data_parameter_dict['rollTest_num'],
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