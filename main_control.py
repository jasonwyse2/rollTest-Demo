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
from your_class import Your_Model, Your_Data

from engine import index_model as im
from data.datalib.slide_Htargets_datasets import get_datasets_2
from data.datalib.slide_Htargets_datasets import get_balanced_shuffled_datasets


if __name__ == '__main__':
    # parameter settings
    # all parameters must be passing by as a dictionary. like: p = Parameters(a = 1, b = 'parameters')
    ############################ prepare data for model to train ##################################
    db_information = {'table': 'index_date'}
    stock_type={'SZ_stock':'1','SH_stock':'2','SZ_index':'100','SH_index':'200'}

    data_parameters_obj = Parameters(
        host='192.168.1.11', port=3306, user='zqfordinary', passwd='Ab123456', db='stock',
        db_table='index_date', db_table_field = 'date, `open`, high, low, `close`, pct_chg, volume, amt ',
        code_wind = '000905', stock_type = stock_type['SH_index'],

        NDays_for_indicatorCalculation = 100,
        train_startTime = '', train_endTime = '20150121', train_lookBack_from_endTime = 1000,
        valid_startTime = '20150121', valid_endTime = '20170101', valid_lookBack_from_endTime = 300,
        test_startTime='20170101', test_endTime='20171231', test_lookBack_from_endTime=100,
        simulativeCloseSeries_directory='/mnt/aidata/QuantitativePlatform/Data/',
        indicator_cutDays = 87,
        dataFile_postfix ='.csv',
        dataType ='train_valid',

        feature_number=10, lookBack=20,
        simulativeCloseSeries_num = 100,

        )

    data_obj = Your_Data(data_parameters_obj)
    data_obj._run()
    ############################ prepare data for model to train ##################################


    ############################  model learning (including training and validation) ##################################

    model_parameters_obj = Parameters(
        data_obj = data_obj,
        modelFile_saveDirectory='/mnt/aidata/QuantitativePlatform/model/',
        train_num=1, batch_size=512*2, nb_classes = 4, nb_input = 10, evaluate_batch_size = 128, evaluate_verbose = 0,
        loss = 'categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_postfix ='.h5',
    )
    model_obj = Your_Model(model_parameters_obj)
    model_obj._run()
    ############################  model learning (including training and validation) ##################################


    ############################  model test ##################################
    # print('### testing ###')
    # test_parameters_obj = Parameters(
    #     model_obj= model_obj,
    #     testFile_saveDirectory='/mnt/aidata/QuantitativePlatform/test/',
    #     dataFile_postfix='.csv',
    #     dataType='test',
    # )
    # test_obj = Your_Test(test_parameters_obj)
    # test_obj._run()
    # model_result = test_obj._args['model']
    # data_dict_test = data_obj._prepare_test_data()
    # test_x, test_y = data_dict_test['test_x'], data_dict_test['test_y']
    # test_obj._test(model_result, test_x, test_y)
    ############################  model test ##################################