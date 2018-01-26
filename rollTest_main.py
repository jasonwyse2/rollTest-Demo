#coding:utf-8
from __future__ import absolute_import
import base_class as base
import numpy as np
import pandas as pd
import tool as tool
import your_class  as your
import os
import sys
import time
from base_class import Parameters
from custom_class import Your_Model, Your_Data

if __name__ == '__main__':
    start = time.time()
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
        train_startTime='20060101', train_endTime='20170201', train_lookBack_from_endTime=0,
        valid_startTime='20170101', valid_endTime='20180101', valid_lookBack_from_endTime=0,
        test_startTime='20170201', test_endTime='20180101', test_lookBack_from_endTime=0,
        train_valid_type = 'train_valid_separate',
        nb_classes = 4,
        project_directory = project_directory,
        dataFile_directoryName =  'data/', ## this is a relative directory, the abslute directory is project_directory+code_wind+dataFile_directoryName
        dataFile_postfix='.csv',
        dataType='train_valid',
        task_description = '2016to2018',

        show_label=False,
        filterTimes_for_upDown = 15,
        filterTimes_for_sharpGentle = 9,
        slope_threshold = 14,

        simulativeCloseSeries_num=1,
        roll_forward_day=30,
    )
    data_parameters_obj._args['test_mostStartTime'] = data_parameters_obj._args['test_startTime']
    data_parameters_obj._args['test_mostEndTime'] = data_parameters_obj._args['test_endTime']
    #test_mostStartTime = '20170101', test_mostEndTime = '20170201',
    model_parameters_obj = Parameters(
        train_num=2, batch_size=512 * 2, nb_classes=4, nb_input=10, evaluate_batch_size=128, evaluate_verbose=0,
        filter_row =2 , filter_col =1, filter_num = 16,
        loss='categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_directoryName = 'models/',## this is a relative directory, the abslute directory is project_directory+code_wind+modelFile_DirectoryName
        modelFile_postfix='',
    )

    train_confuse_df_all, valid_confuse_df_all, test_confuse_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    test_y_true_all, test_y_predict_all = pd.DataFrame(), pd.DataFrame()
    test_filtered_close_all, test_close_all, test_date_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rollTest_num = 1

    for i in range(rollTest_num):
        if i==0:
            roll_forward_day = 0
        else:
            roll_forward_day = data_parameters_obj._args['roll_forward_day']

            data_parameters_obj._args['train_startTime'] = tool.currentDay_forward_delta(
                data_parameters_obj._args['train_startTime'], roll_forward_day)
            data_parameters_obj._args['train_endTime'] = tool.currentDay_forward_delta(
                data_parameters_obj._args['train_endTime'], roll_forward_day)

            data_parameters_obj._args['valid_startTime'] = tool.currentDay_forward_delta(
                data_parameters_obj._args['valid_startTime'], roll_forward_day)
            data_parameters_obj._args['valid_endTime'] = tool.currentDay_forward_delta(
                data_parameters_obj._args['valid_endTime'], roll_forward_day)

            tmpTime = data_parameters_obj._args['test_endTime']
            data_parameters_obj._args['test_endTime'] = tool.currentDay_forward_delta(
                data_parameters_obj._args['test_endTime'], roll_forward_day)
            data_parameters_obj._args['test_startTime'] = tmpTime

            data_parameters_obj._args['test_mostEndTime'] = data_parameters_obj._args['test_endTime']

        data_obj = Your_Data(data_parameters_obj)
        data_obj._run()
        model_parameters_obj._args['data_obj'] = data_obj
        #model_parameters_obj._data_obj = data_obj
        model_obj = Your_Model(model_parameters_obj)

        evaluate_returns, test_returns= model_obj._run()
        train_confuse_df,valid_confuse_df = evaluate_returns[0],evaluate_returns[1]
        test_confuse_df,test_y_true,test_y_predict = test_returns[0], test_returns[1], test_returns[2]
        test_filtered_close, test_close, test_date = test_returns[3], test_returns[4], test_returns[5]

        train_confuse_df_all = pd.concat([train_confuse_df_all, train_confuse_df])
        valid_confuse_df_all = pd.concat([valid_confuse_df_all, valid_confuse_df])
        test_confuse_df_all = pd.concat([test_confuse_df_all, test_confuse_df])

        test_y_true_all = pd.concat([test_y_true_all, pd.DataFrame(test_y_true)])
        test_y_predict_all = pd.concat([test_y_predict_all, pd.DataFrame(test_y_predict)])
        test_filtered_close_all = pd.concat([test_filtered_close_all, pd.DataFrame(test_filtered_close)])
        test_close_all = pd.concat([test_close_all, pd.DataFrame(test_close)])
        test_date_all = pd.concat([test_date_all, pd.DataFrame(test_date)])

    ### save confuse matrix for 'train', 'valid', 'test'
    data_parameter_dict = data_parameters_obj._args
    tool.save_allConfuseMatrix(train_confuse_df_all, data_parameter_dict, type='train')
    tool.save_allConfuseMatrix(valid_confuse_df_all, data_parameter_dict, type='valid')
    tool.save_allConfuseMatrix(test_confuse_df_all, data_parameter_dict, type='test')

    ### merge roll test results into one ###
    data_parameters_dict = data_parameters_obj._args
    data_parameters_dict['dataType'] = 'rollTestAll'
    test_confuse_df_allIntoOne = tool.confuse_matrix(np.array(test_y_true_all), np.array(test_y_predict_all), data_parameters_dict)
    tool.save_allConfuseMatrix(test_confuse_df_allIntoOne, data_parameter_dict, type='rollTestAll')

    ###  save figure results for test ###
    underlyingTime_directory = tool.get_underlyingTime_directory(data_parameter_dict)
    test_mostStartTime = data_parameters_obj._args['test_mostStartTime']
    test_mostEndTime = data_parameters_obj._args['test_mostEndTime']

    test_y_truePredict = [test_y_true_all, test_y_predict_all]
    true_tag_figure_path = underlyingTime_directory + '%s-%s-TruePredictLabels.pdf' % (test_mostStartTime, test_mostEndTime)
    tool.save_figure_pdf_truePredictTogether(test_filtered_close_all, test_close_all, test_date_all, test_y_truePredict, true_tag_figure_path,
                         interval=10)

    elapsed = (time.time() - start)
    print("Time used(minute): ", float(elapsed)/60)