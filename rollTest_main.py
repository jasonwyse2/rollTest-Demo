#coding:utf-8
from __future__ import absolute_import
import base_class as base
import numpy as np
import pandas as pd
import tool as tool
import your_class  as your
import os
import sys
from base_class import Parameters
from custom_class import Your_Model, Your_Data

if __name__ == '__main__':
    # parameter settings
    # all parameters must be passing by as a dictionary. like: p = Parameters(a = 1, b = 'parameters')
    ############################ prepare data for model to train ##################################
    db_information = {'table': 'index_date','host':'192.168.1.11', 'port':3306, 'user':'zqfordinary', 'passwd':'Ab123456', 'db':'stock'}
    stock_type={'SZ_stock':'1','SH_stock':'2','SZ_index':'100','SH_index':'200'}
    project_directory = '/mnt/aidata/QuantitativePlatform/'
    underlying = '000001'
    stock_type = stock_type['SH_index']

    data_parameters_obj = Parameters(
        host=db_information['host'], port=db_information['port'], user=db_information['user'],
        passwd=db_information['passwd'], db=db_information['db'],
        db_table='index_date', db_table_field='date, `open`, high, low, `close`, pct_chg, volume, amt ',
        code_wind=underlying, stock_type=stock_type,

        NTradeDays_for_indicatorCalculation=100,indicator_cutDays = 87,
        extraTradeDays_afterEndTime_for_filter = 2,
        train_startTime='20060101', train_endTime='20150101', train_lookBack_from_endTime=0,
        valid_startTime='20150101', valid_endTime='20160101', valid_lookBack_from_endTime=0,
        test_startTime='20150101', test_endTime='20150201', test_lookBack_from_endTime=0,
        nb_class = 4,
        project_directory = project_directory,
        dataFile_directoryName =  'data/', ## this is a relative directory, the abslute directory is project_directory+code_wind+dataFile_directoryName
        dataFile_postfix='.csv',
        #dataType='train_valid',
        roll_forward_day = 30,

        filter_window_size = 5,
        window_size = 5,
        window_beta = 2,
        kaiser_beta = 2,
        noise_window_size=5,

        show_label=False,
        filter_times_of_direciton_object = 15,
        filter_times_of_gradient_object = 9,
        k_threshold = 14,
        feature_number=10, lookBack=20,
        simulativeCloseSeries_num=2,
    )

    model_parameters_obj = Parameters(
        train_num=2, batch_size=512 * 2, nb_classes=4, nb_input=10, evaluate_batch_size=128, evaluate_verbose=0,
        loss='categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_directoryName = 'models/',## this is a relative directory, the abslute directory is project_directory+code_wind+modelFile_DirectoryName
        modelFile_postfix='',
    )

    rollTest_num = 5
    for i in range(rollTest_num):
        if i==0:
            roll_forward_day = 0
        else:
            roll_forward_day = data_parameters_obj._args['roll_forward_day']

        data_parameters_obj._args['train_startTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['train_startTime'], roll_forward_day)
        data_parameters_obj._args['train_endTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['train_endTime'],roll_forward_day)

        data_parameters_obj._args['valid_startTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['valid_startTime'],roll_forward_day)
        data_parameters_obj._args['valid_endTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['valid_endTime'],roll_forward_day)
        data_parameters_obj._args['test_startTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['test_startTime'],roll_forward_day)
        data_parameters_obj._args['test_endTime'] = tool.currentDay_forward_delta(data_parameters_obj._args['test_endTime'], roll_forward_day)

        data_obj = Your_Data(data_parameters_obj)
        data_obj._run()
        model_parameters_obj._args['data_obj'] = data_obj
        model_obj = Your_Model(model_parameters_obj)
        model_obj._run()


