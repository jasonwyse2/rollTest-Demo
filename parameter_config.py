
stock_type = {'SZ_stock': '1', 'SH_stock': '2', 'SZ_index': '100', 'SH_index': '200'}
day_table = 'index_date'
day_field='date, open, high, low, close, pct_chg, volume, amt '
minute_table = 'index_min'
minute_field='time, open, high, low, close, pct_chg, volume, amt '
#indicator_minute_comb_list = ['day_comb_return_1', 'comb_return_2', 'minute_comb_return_1']
def set_additional_parameters(hyper_parameter_dict, data_parameter_dict):

    config_dayOrMinute_parameter(hyper_parameter_dict, data_parameter_dict)
    config_taskType_parameter(hyper_parameter_dict, data_parameter_dict)

    config_other_parameter(hyper_parameter_dict, data_parameter_dict)


def config_other_parameter(hyper_parameter_dict, data_parameters_dict):
    data_parameters_dict['test_mostStartTime'] = data_parameters_dict['test_startTime']
    data_parameters_dict['test_mostEndTime'] = data_parameters_dict['test_endTime']


    roll_forward = hyper_parameter_dict['roll_forward']
    rollTest_num = hyper_parameter_dict['rollTest_num']

    seq = [hyper_parameter_dict['dayOrMinute'], hyper_parameter_dict['taskType'],
           data_parameters_dict['test_startTime'] + '-' + data_parameters_dict['test_endTime'],
           'RollForward-%s' % str(roll_forward), 'RollNum-%s' % str(rollTest_num)
           ]
    data_parameters_dict['task_description'] = '_'.join(seq)
    windowSize = data_parameters_dict['filter_windowSize']
    data_parameters_dict['extraTradeDays_afterEndTime'] = int(windowSize / 2) if windowSize % 2 == 1 else int(
        windowSize / 2) - 1

def config_dayOrMinute_parameter(hyper_parameter_dict, data_parameter_dict):
    dayOrMinute = hyper_parameter_dict['dayOrMinute']  # dayOrMinute['day', 'minute']
    stock_type = hyper_parameter_dict['stock_type']
    if dayOrMinute=='day' or dayOrMinute=='alpha':
        data_parameter_dict['train_startTime']= hyper_parameter_dict['day_train_startTime']
        data_parameter_dict['train_endTime'] = hyper_parameter_dict['day_train_endTime']
        data_parameter_dict['valid_startTime'] = \
            hyper_parameter_dict['day_train_endTime'] if hyper_parameter_dict['day_valid_startTime']=='' else hyper_parameter_dict['day_valid_startTime']
        data_parameter_dict['valid_endTime'] = hyper_parameter_dict['day_valid_endTime']
        #### test####
        data_parameter_dict['test_startTime'] = \
            hyper_parameter_dict['day_train_endTime'] if hyper_parameter_dict['day_test_startTime']=='' else hyper_parameter_dict['day_test_startTime']
        data_parameter_dict['test_endTime'] =  \
            hyper_parameter_dict['day_valid_endTime'] if hyper_parameter_dict['day_test_endTime']=='' else hyper_parameter_dict['day_test_endTime']
        ### set database 'table' and 'field' ####
        #if stock_type == stock_type['SZ_index'] or stock_type['SH_index']:
        data_parameter_dict['db_table'] = day_table # hyper_parameter_dict['day_table']
        data_parameter_dict['db_field'] = day_field # hyper_parameter_dict['day_field']
        data_parameter_dict['time_field'] = 'date'
        data_parameter_dict['change_threshold'] = hyper_parameter_dict['dalily_change_threshold']

    elif dayOrMinute=='minute_simulative' or dayOrMinute=='minute_no_simulative': #[day, minute_no_simulative, minute_simulative, alpha]
        data_parameter_dict['train_startTime'] = hyper_parameter_dict['minute_train_startTime']
        data_parameter_dict['train_endTime'] = hyper_parameter_dict['minute_train_endTime']
        data_parameter_dict['valid_startTime'] = \
            hyper_parameter_dict['minute_train_endTime'] if hyper_parameter_dict['minute_valid_startTime'] =='' else hyper_parameter_dict['minute_valid_startTime']
        data_parameter_dict['valid_endTime'] = hyper_parameter_dict['minute_valid_endTime']
        #### test####
        data_parameter_dict['test_startTime'] = \
            hyper_parameter_dict['minute_train_endTime'] if hyper_parameter_dict['minute_test_startTime']=='' else hyper_parameter_dict['minute_test_startTime']
        data_parameter_dict['test_endTime'] = \
            hyper_parameter_dict['minute_valid_endTime'] if hyper_parameter_dict['minute_test_endTime']=='' else hyper_parameter_dict['minute_test_endTime']
        ### set database 'table' and 'field' ####
        #if stock_type == stock_type['SZ_index'] or stock_type['SH_index']:
        data_parameter_dict['db_table'] = minute_table
        data_parameter_dict['db_field'] = minute_field
        data_parameter_dict['time_field'] = 'time'
        data_parameter_dict['change_threshold'] = hyper_parameter_dict['minute_change_threshold']
        # train_startTime, train_endTime = data_parameter_dict['minute_train_startTime'], data_parameter_dict['minute_train_endTime']
        # valid_startTime, valid_endTime = data_parameter_dict['minute_valid_startTime'], data_parameter_dict['minute_valid_endTime']
        # test_startTime, test_endTime = data_parameter_dict['minute_test_startTime'], data_parameter_dict['minute_test_endTime']

    else:
        raise Exception('unknown "dayOrMinute":%s'%dayOrMinute)

def config_taskType_parameter(hyper_parameter_dict,data_parameters_dict):
    taskType = hyper_parameter_dict['taskType']  # taskType  ['SharpGentleUpDown','BottomTopUpDown'],
    if taskType == 'SharpGentleUpDown':
        data_parameters_dict['filterTimes_for_upDown'] = 15
        data_parameters_dict['filterTimes_for_sharpGentle'] = 9
        data_parameters_dict['slope_threshold'] = 14
        data_parameters_dict['label_SharpGentleUpDown'] = ['sharp up', 'sharp down', 'gentle up', 'gentle down']
        data_parameters_dict['color_SharpGentleUpDown'] = ['red', 'green', 'violet', 'lightgreen']

    elif taskType == 'BottomTopUpDown':
        data_parameters_dict['kneeNum_at_bottomTop'] = 2
        data_parameters_dict['filterTimes_for_upDown'] = 1
        data_parameters_dict['change_percent'] = 0.015
        data_parameters_dict['label_BottomTopUpDown'] = ['bottom', 'up', 'top', 'down']
        data_parameters_dict['color_BottomTopUpDown'] = ['green', 'violet', 'red', 'lightgreen']
    else:
        raise Exception('unknown "taskType":%s' % taskType)

