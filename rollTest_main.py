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
    underlying = '000905'
    stock_type = stock_type['SH_index']

    data_parameters_obj = Parameters(
        host=db_information['host'], port=db_information['port'], user=db_information['user'],
        passwd=db_information['passwd'], db=db_information['db'],
        db_table='index_date', db_table_field='date, open, high, low, close, pct_chg, volume, amt ',
        code_wind=underlying, stock_type=stock_type,

        NTradeDays_for_indicatorCalculation=100,
        filter_windowSize=5,
        kaiser_beta=2,
        addNoise_windowSize=5,
        extraTradeDays_afterEndTime_for_filter = 2,
        extraTradeDays_beforeStartTime_for_filter=2,
        train_startTime='20060101', train_endTime='20150101', train_lookBack_from_endTime=0,
        valid_startTime='20150101', valid_endTime='20160101', valid_lookBack_from_endTime=0,
        test_startTime='20150101', test_endTime='20150201', test_lookBack_from_endTime=0,
        nb_class = 4,
        project_directory = project_directory,
        dataFile_directoryName =  'data/', ## this is a relative directory, the abslute directory is project_directory+code_wind+dataFile_directoryName
        dataFile_postfix='.csv',
        dataType='train_valid',

        show_label=False,
        filterTimes_for_upDown = 15,
        filterTimes_for_sharpGentle = 9,
        slope_threshold = 14,

        simulativeCloseSeries_num=10,
        roll_forward_day=30,
    )

    model_parameters_obj = Parameters(
        train_epoch=2, batch_size=512 * 2, nb_classes=4, nb_input=10, evaluate_batch_size=128, evaluate_verbose=0,
        filter_row =2 , filter_col =1, filter_num = 16,
        loss='categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_directoryName = 'models/',## this is a relative directory, the abslute directory is project_directory+code_wind+modelFile_DirectoryName
        modelFile_postfix='.h5',
    )

    train_confu_matrix_df_all, valid_confu_matrix_df_all, test_confu_matrix_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rollTest_num = 2
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
        train_confu_matrix_df, valid_confu_matrix_df, test_confu_matrix_df = model_obj._run()
        train_confu_matrix_df_all = pd.concat([train_confu_matrix_df_all,train_confu_matrix_df])
        valid_confu_matrix_df_all = pd.concat([valid_confu_matrix_df_all,valid_confu_matrix_df])
        test_confu_matrix_df_all = pd.concat([test_confu_matrix_df_all,test_confu_matrix_df])

        if(i==rollTest_num-1):
            parameter_dict = data_obj._data_parameters_obj._args
            tool.save_allConfuseMatrix(train_confu_matrix_df_all, parameter_dict, type ='train')
            tool.save_allConfuseMatrix(train_confu_matrix_df_all, parameter_dict, type ='valid')
            tool.save_allConfuseMatrix(train_confu_matrix_df_all, parameter_dict, type ='test')
    print train_confu_matrix_df_all
