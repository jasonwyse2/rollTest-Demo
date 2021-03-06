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
def get_roll_start(data_parameter_obj, model_parameter_obj):
    #data_parameter_obj = model_parameter_obj._args['data_parameter_obj']
    rollTest_num = model_parameter_obj._args['rollTest_num']
    train_confuse_df_all, valid_confuse_df_all, test_confuse_df_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    test_y_true_all, test_y_predict_all = pd.DataFrame(), pd.DataFrame()
    test_filtered_close_all, test_close_all, test_date_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i in range(rollTest_num):
        print('i=%s' % str(i))
        if i == 0:
            pass
        else:
            roll_forward = data_parameter_obj._args['roll_forward']
            data_parameter_dict = data_parameter_obj._args
            # data_parameter_obj._args['train_startTime'] = tool.currentDay_forward_delta(
            #     data_parameter_obj._args['train_startTime'], roll_forward)
            data_parameter_obj._args['train_endTime'] = tool.currentTime_forward_delta(
                data_parameter_obj._args['train_endTime'], roll_forward, data_parameter_dict)

            data_parameter_obj._args['valid_startTime'] = tool.currentTime_forward_delta(
                data_parameter_obj._args['valid_startTime'], roll_forward, data_parameter_dict)
            data_parameter_obj._args['valid_endTime'] = tool.currentTime_forward_delta(
                data_parameter_obj._args['valid_endTime'], roll_forward, data_parameter_dict)

            tmpTime = data_parameter_obj._args['test_endTime']
            data_parameter_obj._args['test_endTime'] = tool.currentTime_forward_delta(
                data_parameter_obj._args['test_endTime'], roll_forward, data_parameter_dict)
            data_parameter_obj._args['test_startTime'] = tmpTime

            data_parameter_obj._args['test_mostEndTime'] = data_parameter_obj._args['test_endTime']

        data_obj = Your_Data(data_parameter_obj)
        data_obj._run()

        model_parameter_obj._args['data_obj'] = data_obj
        model_obj = Your_Model(model_parameter_obj)
        evaluate_return_list, test_return_list = model_obj._run()

        # parse results into separate variables
        train_confuse_df, valid_confuse_df = evaluate_return_list[0], evaluate_return_list[1]
        test_confuse_df, test_y_true, test_y_predict = test_return_list[0], test_return_list[1], test_return_list[2]
        test_filtered_close, test_close, test_date = test_return_list[3], test_return_list[4], test_return_list[5]
        # concatenate confuse Dataframe into an entire one
        train_confuse_df_all = pd.concat([train_confuse_df_all, train_confuse_df])
        valid_confuse_df_all = pd.concat([valid_confuse_df_all, valid_confuse_df])
        test_confuse_df_all = pd.concat([test_confuse_df_all, test_confuse_df])

        test_y_true_all = pd.concat([test_y_true_all, pd.DataFrame(test_y_true)])
        test_y_predict_all = pd.concat([test_y_predict_all, pd.DataFrame(test_y_predict)])
        test_filtered_close_all = pd.concat([test_filtered_close_all, pd.DataFrame(test_filtered_close)])
        test_close_all = pd.concat([test_close_all, pd.DataFrame(test_close)])
        test_date_all = pd.concat([test_date_all, pd.DataFrame(test_date)])


    roll_result_dict={'train_confuse_df_all':train_confuse_df_all, 'valid_confuse_df_all':valid_confuse_df_all,
               'test_confuse_df_all':test_confuse_df_all,
                'test_y_true_all':test_y_true_all, 'test_y_predict_all':test_y_predict_all,
               'test_filtered_close_all':test_filtered_close_all, 'test_close_all':test_close_all,
               'test_date_all':test_date_all}
    return roll_result_dict