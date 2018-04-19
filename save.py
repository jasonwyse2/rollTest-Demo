import tool
import numpy as np
import json
import net
def save_file(data_parameter_obj, roll_result_dict):
    train_confuse_df_all = roll_result_dict['train_confuse_df_all']
    valid_confuse_df_all = roll_result_dict['valid_confuse_df_all']
    test_confuse_df_all = roll_result_dict['test_confuse_df_all']
    test_y_true_all = roll_result_dict['test_y_true_all']
    test_y_predict_all = roll_result_dict['test_y_predict_all']
    test_close_all = roll_result_dict['test_close_all']
    test_date_all = roll_result_dict['test_date_all']

    ### save confuse matrix for 'train', 'valid', 'test'
    data_parameter_dict = data_parameter_obj._args
    tool.save_allConfuseMatrix(train_confuse_df_all, data_parameter_dict, dataType='train')
    tool.save_allConfuseMatrix(valid_confuse_df_all, data_parameter_dict, dataType='valid')
    tool.save_allConfuseMatrix(test_confuse_df_all, data_parameter_dict, dataType='test')

    ### merge roll test results into one ###
    test_confuse_df_allIntoOne = tool.confuse_matrix(np.array(test_y_true_all), np.array(test_y_predict_all),
                                                     data_parameter_dict)
    tool.save_allConfuseMatrix(test_confuse_df_allIntoOne, data_parameter_dict, dataType='AllTest')

    ## save predict results for return calculation
    parameter_directory = tool.get_underlyingTime_directory(data_parameter_dict)
    result_fileName = parameter_directory + 'netRelatedVariables.json'
    file_obj = open(result_fileName, 'w')
    save_result_dict = {'test_y_true_all': [v[0] for v in np.array(test_y_true_all)],
                        'test_y_predict_all': [v[0] for v in np.array(test_y_predict_all)],
                        'test_close_all': [v[0] for v in np.array(test_close_all)],
                        'test_date_all': [v[0] for v in np.array(test_date_all)]}
    file_obj.write(json.dumps(save_result_dict))
    file_obj.close()

    file_read_obj = open(result_fileName, 'r')
    parameter_file_write = parameter_directory + 'easyToRead_netRelatedParameter.txt'
    file_write_obj = open(parameter_file_write, 'w')
    line = file_read_obj.readline()
    items = line.split(':')
    for item in items:
        file_write_obj.write(item)
        file_write_obj.write('\n')

    interval = data_parameter_dict['interval']
    net.get_net_yield(save_result_dict, parameter_directory, data_parameter_dict['code_wind'],interval = interval)

def save_figure(data_parameter_obj, roll_result_dict):
    ###  save figure results for test ###
    data_parameter_dict = data_parameter_obj._args
    underlyingTime_directory = tool.get_underlyingTime_directory(data_parameter_dict)
    test_mostStartTime = data_parameter_dict['test_mostStartTime']
    test_mostEndTime = data_parameter_dict['test_mostEndTime']
    code_wind = data_parameter_dict['code_wind']

    test_y_true_all = roll_result_dict['test_y_true_all']
    test_y_predict_all = roll_result_dict['test_y_predict_all']
    test_close_all = roll_result_dict['test_close_all']
    test_date_all = roll_result_dict['test_date_all']
    test_filtered_close_all = roll_result_dict['test_filtered_close_all']

    test_y_TruePredict = [test_y_true_all, test_y_predict_all]
    true_tag_figure_path = underlyingTime_directory + '%s-%s-%s-TruePredictLabels.pdf' % (code_wind,
    test_mostStartTime, test_mostEndTime)

    fourlabelType = data_parameter_dict['taskType']
    if fourlabelType == 'SharpGentleUpDown':
        color_list = data_parameter_dict['color_SharpGentleUpDown']
        label_list = data_parameter_dict['label_SharpGentleUpDown']
    elif fourlabelType == 'BottomTopUpDown':
        color_list = data_parameter_dict['color_BottomTopUpDown']
        label_list = data_parameter_dict['label_BottomTopUpDown']
    elif fourlabelType == 'Volatility':
        color_list = data_parameter_dict['color_Volatility']
        label_list = data_parameter_dict['label_Volatility']
    interval = data_parameter_dict['interval']
    tool.save_figure_pdf_truePredictTogether(test_filtered_close_all, test_close_all, test_date_all, test_y_TruePredict,
                                             true_tag_figure_path, code_wind, color=color_list, label=label_list, interval=interval)