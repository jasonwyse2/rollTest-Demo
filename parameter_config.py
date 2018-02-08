
stock_type = {'SZ_stock': '1', 'SH_stock': '2', 'SZ_index': '100', 'SH_index': '200'}
day_table = 'index_date'
day_field='date, open, high, low, close, pct_chg, volume, amt '
minute_table = 'index_min'
minute_field='time, open, high, low, close, pct_chg, volume, amt '
minuteType_dict = {'minuteSimulative': 'minuteSimulative', 'minuteNoSimulative': 'minuteNoSimulative'}
interval_dict = {'dayInterval':60, 'minuteInterval':40}
#indicator_minute_comb_list = ['day_comb_return_1', 'comb_return_2', 'minute_comb_return_1']
def set_additional_parameters(hyper_parameter_dict, data_parameter_dict):

    config_dayOrMinute_parameter(hyper_parameter_dict, data_parameter_dict)
    config_taskType_parameter(hyper_parameter_dict, data_parameter_dict)

    config_other_parameter(hyper_parameter_dict, data_parameter_dict)

def config_dayOrMinute_parameter(hyper_parameter_dict, data_parameter_dict):
    dayOrMinute = hyper_parameter_dict['dayOrMinute']  # dayOrMinute['day', 'minute']
    data_parameter_dict['minuteType_dict'] = minuteType_dict
    if dayOrMinute=='alpha':
        data_parameter_dict['code_wind'] = 'none'
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
        data_parameter_dict['db_table'] = day_table # hyper_parameter_dict['day_table']
        data_parameter_dict['db_field'] = day_field # hyper_parameter_dict['day_field']
        data_parameter_dict['time_field'] = 'date'
        data_parameter_dict['change_threshold'] = hyper_parameter_dict['dalily_change_threshold']
        data_parameter_dict['interval'] = interval_dict['dayInterval']
        data_parameter_dict['roll_forward'] = hyper_parameter_dict['day_roll_forward']
        data_parameter_dict['rollTest_num'] = hyper_parameter_dict['day_rollTest_num']

    elif dayOrMinute in minuteType_dict:

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
        data_parameter_dict['db_table'] = minute_table
        data_parameter_dict['db_field'] = minute_field
        data_parameter_dict['time_field'] = 'time'
        data_parameter_dict['change_threshold'] = hyper_parameter_dict['minute_change_threshold']
        data_parameter_dict['interval'] = interval_dict['dayInterval']
        data_parameter_dict['roll_forward'] = hyper_parameter_dict['minute_roll_forward']
        data_parameter_dict['rollTest_num'] = hyper_parameter_dict['minute_rollTest_num']

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
        data_parameters_dict['filterTimes_for_upDown'] = 0
        data_parameters_dict['label_BottomTopUpDown'] = ['bottom', 'up', 'top', 'down']
        data_parameters_dict['color_BottomTopUpDown'] = ['green', 'violet', 'red', 'lightgreen']
    else:
        raise Exception('unknown "taskType":%s' % taskType)

def config_other_parameter(hyper_parameter_dict, data_parameters_dict):
    data_parameters_dict['test_mostStartTime'] = data_parameters_dict['test_startTime']
    data_parameters_dict['test_mostEndTime'] = data_parameters_dict['test_endTime']

    roll_forward = data_parameters_dict['roll_forward']
    rollTest_num = data_parameters_dict['rollTest_num']

    seq = [hyper_parameter_dict['dayOrMinute'], hyper_parameter_dict['taskType'],
           data_parameters_dict['test_startTime'] + '-' + data_parameters_dict['test_endTime'],
           'roll-%s-%s' % (str(roll_forward),str(rollTest_num))
           ]
    data_parameters_dict['task_description'] = '_'.join(seq)
    windowSize = data_parameters_dict['filter_windowSize']
    data_parameters_dict['extraTradeDays_afterEndTime'] = int(windowSize / 2) if windowSize % 2 == 1 else int(
        windowSize / 2) - 1
if __name__ == '__main__':
    minuteType_dict = {'a': 'ab', 'b': 'bc'}
    print 'ab' in minuteType_dict

