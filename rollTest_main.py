#coding:utf-8
from __future__ import absolute_import
import numpy as np
import pandas as pd
import tool as tool
import time
from base_class import Parameters
from custom_class import Your_Model, Your_Data
import json
from returnCurve import get_return
import matplotlib.pyplot as plt
import matplotlib
from roll import get_roll_start
from save import save_figure,save_file
if __name__ == '__main__':
    start = time.time()
    # parameter settings
    # all parameters must be passing by as a dictionary. like: p = Parameters(a = 1, b = 'parameters')
    ############################ prepare data for model to train ##################################
    db_information = {'table': 'index_date','host':'192.168.1.11', 'port':3306, 'user':'zqfordinary', 'passwd':'Ab123456', 'db':'stock'}
    stock_type={'SZ_stock':'1','SH_stock':'2','SZ_index':'100','SH_index':'200'}
    project_directory = '/mnt/aidata/QuantitativePlatform/a-yaogong/'
    underlying = '000905'
    stock_type = stock_type['SH_index']
    dayOrMinute = 'day' # 'minute'
    data_parameters_obj = Parameters(
        host=db_information['host'], port=db_information['port'], user=db_information['user'],
        passwd=db_information['passwd'], db=db_information['db'],
        db_table='index_date', db_table_field='date, open, high, low, close, pct_chg, volume, amt ',
        code_wind=underlying, stock_type=stock_type,

        NTradeDays_for_indicatorCalculation=100,
        filter_windowSize=5,
        kaiser_beta=2,
        addNoise_windowSize=5,

        #extraTradeDays_beforeStartTime_for_filter=2,
        train_startTime='20060101', train_endTime='20150101', train_lookBack_from_endTime=0,
        valid_startTime='20150101', valid_endTime='20150201', valid_lookBack_from_endTime=0,
        test_startTime='20150101', test_endTime='20150201', test_lookBack_from_endTime=0,
        train_valid_type = 'train_valid_separate',
        nb_classes = 4,
        project_directory = project_directory,
        dataFile_directoryName =  'data/', ## this is a relative directory, the abslute directory is project_directory+code_wind+dataFile_directoryName
        dataFile_postfix='.csv',
        dataType='train',
        task_description = 'yaogong-2015allyear-',
        fourlabelType = 'BottomTopUpDown', change_percent = 0.015,# label_type = 'BottomTopUpDown' , 'SharpGentleUpDown'
        kneeNum_at_bottomTop = 2,
        show_label=False,
        filterTimes_for_upDown = 1,
        filterTimes_for_sharpGentle = 15,
        slope_threshold = 14,

        simulativeCloseSeries_num=1,
        roll_forward_day=30,
    )
    data_parameters_obj._args['test_mostStartTime'] = data_parameters_obj._args['test_startTime']
    data_parameters_obj._args['test_mostEndTime'] = data_parameters_obj._args['test_endTime']
    data_parameters_obj._args['task_description'] = data_parameters_obj._args['task_description']+data_parameters_obj._args['fourlabelType']

    model_parameters_obj = Parameters(
        train_num=1, rollTest_num = 1,

        batch_size=512 * 2, nb_classes=4, nb_input=10, evaluate_batch_size=128, evaluate_verbose=0,
        filter_row =2 , filter_col =1, filter_num = 16,
        loss='categorical_crossentropy', optimizer='rmsprop', loss_weights='loss_weights', metrics=['accuracy'],
        modelFile_directoryName = 'models/',## this is a relative directory, the abslute directory is project_directory+code_wind+modelFile_DirectoryName
        modelFile_postfix='',
    )

    roll_result_dict = get_roll_start(data_parameters_obj, model_parameters_obj)

    # rollTest_num = model_parameters_obj._args['rollTest_num']
    # train_confuse_df_all, valid_confuse_df_all, test_confuse_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # test_y_true_all, test_y_predict_all = pd.DataFrame(), pd.DataFrame()
    # test_filtered_close_all, test_close_all, test_date_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #
    # for i in range(rollTest_num):
    #     print('i=%s'%str(i))
    #     if i==0:
    #         roll_forward_day = 0
    #     else:
    #         roll_forward_day = data_parameters_obj._args['roll_forward_day']
    #
    #         data_parameters_obj._args['train_startTime'] = tool.currentDay_forward_delta(
    #             data_parameters_obj._args['train_startTime'], roll_forward_day)
    #         data_parameters_obj._args['train_endTime'] = tool.currentDay_forward_delta(
    #             data_parameters_obj._args['train_endTime'], roll_forward_day)
    #
    #         data_parameters_obj._args['valid_startTime'] = tool.currentDay_forward_delta(
    #             data_parameters_obj._args['valid_startTime'], roll_forward_day)
    #         data_parameters_obj._args['valid_endTime'] = tool.currentDay_forward_delta(
    #             data_parameters_obj._args['valid_endTime'], roll_forward_day)
    #
    #         tmpTime = data_parameters_obj._args['test_endTime']
    #         data_parameters_obj._args['test_endTime'] = tool.currentDay_forward_delta(
    #             data_parameters_obj._args['test_endTime'], roll_forward_day)
    #         data_parameters_obj._args['test_startTime'] = tmpTime
    #
    #         data_parameters_obj._args['test_mostEndTime'] = data_parameters_obj._args['test_endTime']
    #
    #     data_obj = Your_Data(data_parameters_obj)
    #     data_obj._run()
    #     model_parameters_obj._args['data_obj'] = data_obj
    #     #model_parameters_obj._data_obj = data_obj
    #     model_obj = Your_Model(model_parameters_obj)
    #     evaluate_returns, test_returns= model_obj._run()
    #
    #     train_confuse_df,valid_confuse_df = evaluate_returns[0],evaluate_returns[1]
    #     test_confuse_df,test_y_true,test_y_predict = test_returns[0], test_returns[1], test_returns[2]
    #     test_filtered_close, test_close, test_date = test_returns[3], test_returns[4], test_returns[5]
    #
    #     train_confuse_df_all = pd.concat([train_confuse_df_all, train_confuse_df])
    #     valid_confuse_df_all = pd.concat([valid_confuse_df_all, valid_confuse_df])
    #     test_confuse_df_all = pd.concat([test_confuse_df_all, test_confuse_df])
    #
    #     test_y_true_all = pd.concat([test_y_true_all, pd.DataFrame(test_y_true)])
    #     test_y_predict_all = pd.concat([test_y_predict_all, pd.DataFrame(test_y_predict)])
    #     test_filtered_close_all = pd.concat([test_filtered_close_all, pd.DataFrame(test_filtered_close)])
    #     test_close_all = pd.concat([test_close_all, pd.DataFrame(test_close)])
    #     test_date_all = pd.concat([test_date_all, pd.DataFrame(test_date)])

    ### save confuse matrix for 'train', 'valid', 'test'
    save_file(data_parameters_obj, roll_result_dict)
    # data_parameters_dict = data_parameters_obj._args
    # tool.save_allConfuseMatrix(train_confuse_df_all, data_parameters_dict, type='train')
    # tool.save_allConfuseMatrix(valid_confuse_df_all, data_parameters_dict, type='valid')
    # tool.save_allConfuseMatrix(test_confuse_df_all, data_parameters_dict, type='test')
    #
    # ### merge roll test results into one ###
    # #data_parameters_dict = data_parameters_obj._args
    # data_parameters_dict['dataType'] = 'rollTestAll'
    # test_confuse_df_allIntoOne = tool.confuse_matrix(np.array(test_y_true_all), np.array(test_y_predict_all), data_parameters_dict)
    # tool.save_allConfuseMatrix(test_confuse_df_allIntoOne, data_parameters_dict, type='AllTest')
    #
    # ## save predict results for return calculation
    # parameter_directory = tool.get_underlyingTime_directory(data_parameters_dict)
    # result_fileName = parameter_directory + 'result.json'
    # file_obj = open(result_fileName, 'w')
    # save_result_dict = {'test_y_true_all': [v[0] for v in  np.array(test_y_true_all)], 'test_y_predict_all': [v[0] for v in  np.array(test_y_predict_all)],
    #                     'test_close_all': [v[0] for v in  np.array(test_close_all)], 'test_date_all': [v[0] for v in  np.array(test_date_all)]}
    # file_obj.write(json.dumps(save_result_dict))
    # file_obj.close()

    #asset, amount = get_return(save_result_dict,parameter_directory)

    ###  save figure results for test ###
    save_figure(data_parameters_obj, roll_result_dict)
    # underlyingTime_directory = tool.get_underlyingTime_directory(data_parameters_dict)
    # test_mostStartTime = data_parameters_obj._args['test_mostStartTime']
    # test_mostEndTime = data_parameters_obj._args['test_mostEndTime']
    #
    # test_y_TruePredict = [test_y_true_all, test_y_predict_all]
    # true_tag_figure_path = underlyingTime_directory + '%s-%s-%s-TruePredictLabels.pdf' % (underlying, test_mostStartTime, test_mostEndTime)
    # label1 = ['sharp up', 'sharp down', 'gentle up', 'gentle down']
    # color1 = ['red', 'green', 'violet', 'lightgreen']
    # label2 = ['bottom', 'up', 'top', 'down']
    # color2 = ['green', 'violet', 'red', 'lightgreen']
    # fourlabelType = data_parameters_dict['fourlabelType']
    # if fourlabelType == 'SharpGentleUpDown':
    #     color_list = color1
    #     label_list = label1
    # elif fourlabelType == 'BottomTopUpDown':
    #     color_list = color2
    #     label_list = label2
    #
    # tool.save_figure_pdf_truePredictTogether(test_filtered_close_all, test_close_all, test_date_all, test_y_TruePredict,
    #                                          true_tag_figure_path, underlying, color = color_list, label = label_list, interval=10)
    #



    elapsed = (time.time() - start)
    print("Time used(minute): ", float(elapsed)/60)

    #test_close_all, test_date_all, test_y_TruePredict