#coding:utf-8
import base_class as base
import numpy as np
import pandas as pd
import os
from data.load_data import get_datasets_2, get_x_y
from data.load_data import get_balanced_shuffled_datasets
from keras.utils.np_utils import to_categorical
from keras.layers import Input, BatchNormalization,Dropout,Convolution2D, Dense,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.models import Sequential,load_model,Model
import tool as tool
from engine import index_model as im
import keras
import json

class Your_Data(base.Data):
    '''
        write your own data process
    '''
    def __init__(self, data_parameters_obj):
        self._data_parameters_obj = data_parameters_obj
        self.__checkLegalInput()

    def __checkLegalInput(self):
        #### check legality for train data ###
        train_endTime = self._data_parameters_obj._args['train_endTime']
        if train_endTime=='':
            raise Exception('"train_endTime" can not be empty')
        train_startTime = self._data_parameters_obj._args['train_startTime']
        train_lookBack_from_endTime = self._data_parameters_obj._args['train_lookBack_from_endTime']
        # if not train_startTime=='' and not train_lookBack_from_endTime==0:
        #     raise Exception('"train_startTime" and "train_lookBack_from_endTime" can not be assigned simultaneously')
        # #### check legality for valid data ###
        # valid_endTime = self._data_parameters_obj._args['valid_endTime']
        # if valid_endTime == '':
        #     raise Exception('"valid_endTime" can not be empty')
        # valid_startTime = self._data_parameters_obj._args['valid_startTime']
        # valid_lookBack_from_endTime = self._data_parameters_obj._args['valid_lookBack_from_endTime']
        # if not valid_startTime == '' and not valid_lookBack_from_endTime == 0:
        #     raise Exception('"valid_startTime" and "valid_lookBack_from_endTime" can not be assigned simultaneously')
        #### check legality for test data ###
        test_endTime = self._data_parameters_obj._args['test_endTime']
        if test_endTime == '':
            raise Exception('"test_endTime" can not be empty')
        test_startTime = self._data_parameters_obj._args['test_startTime']
        test_lookBack_from_endTime = self._data_parameters_obj._args['valid_lookBack_from_endTime']
        # if not test_startTime == '' and not test_lookBack_from_endTime == 0:
        #     raise Exception(
        #         '"test_startTime" and "test_lookBack_from_endTime" can not be assigned simultaneously')

    def _get_dataFileName(self):
        parameter_dict = self._data_parameters_obj._args
        file_name = tool.get_onlyFileName(parameter_dict)

        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        abslute_dataFile_directory = underlyingTime_directory + parameter_dict['dataFile_directoryName']
        tool.make_directory(abslute_dataFile_directory)

        postfix = parameter_dict['dataFile_postfix']
        file_FullName = abslute_dataFile_directory + file_name + postfix
        return file_FullName, file_name

    def _run(self):
        self._prepare_trainValid_data2()

    def __read_mat_from_csv(self,csv_path):
        train_start_time = self._data_parameters_obj._args['train_start_time']
        train_end_time = self._data_parameters_obj._args['train_end_time']
        raw_df = pd.read_csv(csv_path)
        date_list = np.array(raw_df.date.tolist()).astype(np.str)
        start_idx = np.where(date_list >= train_start_time)[0][0]
        end_idx = np.where(date_list <= train_end_time)[0][-1]
        raw_sample = raw_df.iloc[:, 1:].as_matrix()
        raw_sample_mat = raw_sample[start_idx:end_idx + 1, :]
        return raw_sample_mat

    # def __split_train_valid(self, raw_sample_mat):
    #     '''
    #     it contains cutting the extra days for tagging data
    #     :param raw_sample_mat:
    #     :param lookBack_from_validEndTime:
    #     :return:
    #     '''
    #     valid_lookBack_from_endTime = self._data_parameters_obj._args['valid_lookBack_from_endTime']
    #
    #     extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
    #
    #     train_x_list, train_y_list = [], []
    #     valid_x_list, valid_y_list = [], []
    #     closeSeries_num = raw_sample_mat.shape[1]
    #     show_label = self._data_parameters_obj._args['show_label']
    #     for i in range(closeSeries_num):
    #         close_array = raw_sample_mat[:, i]
    #         close = pd.Series(close_array)
    #         x, y, filtered_close_for_use, close_for_use = get_datasets_2(close, self._data_parameters_obj._args,show_label=show_label)
    #
    #         train_x, train_y = x[:-(valid_lookBack_from_endTime+extraTradeDays)], y[:-(valid_lookBack_from_endTime+extraTradeDays)]
    #         valid_x, valid_y = x[-(valid_lookBack_from_endTime+extraTradeDays):-extraTradeDays],\
    #                            y[-(valid_lookBack_from_endTime+extraTradeDays):-extraTradeDays]
    #         train_x_list.append(train_x)
    #         train_y_list.append(train_y)
    #         valid_x_list.append(valid_x)
    #         valid_y_list.append(valid_y)
    #     return train_x_list, train_y_list, valid_x_list, valid_y_list
    # def _prepare_test_data(self):
    #     self._data_parameters_obj._args['dataType'] = 'test'
    #     df = tool.get_daily_data(self._data_parameters_obj._args)
    #     close_array = np.array(df['close'].tolist())
    #     close = pd.Series(close_array)
    #     # pct = np.diff(close_array) / close_array[:-1]
    #     self._test_x, self._test_y, self._test_filtered_close_for_use, self._test_close_for_use = get_datasets_2(close,self._data_parameters_obj._args,show_label=False)
    #     ### cut extra days for tagging data, in order to keep consistent with date given in 'data_parameters_obj'
    #     extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
    #     self._test_x, self._test_y,  = self._test_x[:-extraTradeDays], self._test_y[:-extraTradeDays]
    #     self._test_filtered_close_for_use, self._test_close_for_use = self._test_filtered_close_for_use[:-extraTradeDays], self._test_close_for_use[:-extraTradeDays]
    #     self._test_x = self._test_x.reshape(-1, self._test_x.shape[1], 1, 1)

    def _prepare_data(self,dataType=''):
        self._data_parameters_obj._args['dataType'] = dataType
        show_label = self._data_parameters_obj._args['show_label']
        raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)
        # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
        simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, self._data_parameters_obj._args)
        datafile_fullName, datafile_name = self._get_dataFileName()
        if not os.path.exists(datafile_fullName):
            simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
        simulative_sample_matrix = simulativeCloseSeries_df.as_matrix()

        closeSeries_num = simulative_sample_matrix.shape[1]
        x_list, y_list = [], []
        filtered_close_list, close_list = [], []
        for i in range(closeSeries_num):
            close_array = simulative_sample_matrix[:, i]
            close = pd.Series(close_array)
            x, y, filtered_close_for_use, close_for_use = get_x_y(close, self._data_parameters_obj._args)
            if show_label == True:
                tool.show_fig(y, filtered_close_for_use, close_for_use)
            x_list.append(x)
            y_list.append(y)
            filtered_close_list.append(filtered_close_for_use), close_list
        return x_list, y_list


    def _prepare_test_data(self):
        self._data_parameters_obj._args['dataType'] = 'test'
        df = tool.get_daily_data(self._data_parameters_obj._args)
        extra_front = self._data_parameters_obj._args['NTradeDays_for_indicatorCalculation']
        extra_end = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
        self._test_x_date = df.date[extra_front:-extra_end]
        close_array = np.array(df['close'].tolist())
        close = pd.Series(close_array)
        # pct = np.diff(close_array) / close_array[:-1]
        x, y, filtered_close_for_use, close_for_use = get_x_y(close, self._data_parameters_obj._args)
        self._test_x, self._test_y = x, y
        self._test_filtered_close_for_use, self._test_close_for_use = filtered_close_for_use, close_for_use

        self._test_x = self._test_x.reshape(-1, self._test_x.shape[1], 1, 1)
    # def _prepare_train_data(self,dataType=''):
    #     self._data_parameters_obj._args['dataType'] = 'train'
    #     raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)
    #     # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
    #     simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, self._data_parameters_obj._args)
    #     datafile_fullName, datafile_name = self._get_dataFileName()
    #     if not os.path.exists(datafile_fullName):
    #         simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
    #     simulative_sample_matrix = simulativeCloseSeries_df.as_matrix()
    #
    #     closeSeries_num = simulative_sample_matrix.shape[1]
    #     train_x_list, train_y_list = [], []
    #     extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
    #     for i in range(closeSeries_num):
    #         close_array = simulative_sample_matrix[:, i]
    #         close = pd.Series(close_array)
    #         x, y, filtered_close_for_use, close_for_use = get_x_y(close, self._data_parameters_obj._args)
    #         ## show the figure
    #         #tool.show_fig(y, filtered_close_for_use, close_for_use)
    #         #train_x, train_y = x[:-(extraTradeDays)], y[:-(extraTradeDays)]
    #         train_x_list.append(x)
    #         train_y_list.append(y)
    #     return train_x_list, train_y_list
    #
    # def _prepare_valid_data(self, dataType=''):
    #     self._data_parameters_obj._args['dataType'] = 'valid'
    #     raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)
    #     # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
    #     simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, self._data_parameters_obj._args)
    #     datafile_fullName, datafile_name = self._get_dataFileName()
    #     if not os.path.exists(datafile_fullName):
    #         simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
    #     raw_sample_matrix = simulativeCloseSeries_df.as_matrix()
    #
    #     closeSeries_num = raw_sample_matrix.shape[1]
    #     valid_x_list, valid_y_list = [], []
    #     extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
    #     for i in range(closeSeries_num):
    #         close_array = raw_sample_matrix[:, i]
    #         close = pd.Series(close_array)
    #         x, y, filtered_close_for_use, close_for_use = get_x_y(close, self._data_parameters_obj._args)
    #         ## show the figure
    #         # tool.show_fig(y, filtered_close_for_use, close_for_use)
    #         valid_x_list.append(x)
    #         valid_y_list.append(y)
    #     return valid_x_list, valid_y_list

    def _prepare_trainValid_data2(self):
        train_x_list, train_y_list = self._prepare_data(dataType='train')
        valid_x_list, valid_y_list = self._prepare_data(dataType='valid')
        train_x_ndarray, train_x_label_ndarray = np.row_stack(train_x_list), np.hstack(train_y_list)
        valid_x_ndarray, valid_x_label_ndarray = np.row_stack(valid_x_list), np.hstack(valid_y_list)

        print 'train_x_ndarray', train_x_ndarray.shape
        parameter_dict = self._data_parameters_obj._args
        balancedShuffled_train_x, balancedShuffled_train_y = \
            get_balanced_shuffled_datasets(train_x_ndarray, train_x_label_ndarray, parameter_dict)
        print 'balancedShuffled_train_x.shape', balancedShuffled_train_x.shape
        #balancedShuffled_valid_x, balancedShuffled_valid_y = \
        #    get_balanced_shuffled_datasets(valid_x_ndarray, valid_x_label_ndarray, data_parameter_dict)
        balancedShuffled_valid_x = valid_x_ndarray.reshape(-1, valid_x_ndarray.shape[1], 1, 1)
        balancedShuffled_valid_y = valid_x_label_ndarray
        #balancedShuffled_valid_x, balancedShuffled_valid_y = valid_x_ndarray, valid_x_label_ndarray
        print 'balancedShuffled_valid_x.shape,', balancedShuffled_valid_x.shape
        self._train_x, self._train_y = balancedShuffled_train_x, balancedShuffled_train_y
        self._valid_x, self._valid_y = balancedShuffled_valid_x, balancedShuffled_valid_y


    # def _prepare_trainValid_data(self):
    #     self._data_parameters_obj._args['dataType'] = 'train_valid'
    #     raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)
    #     # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
    #     simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, self._data_parameters_obj._args)
    #     datafile_fullName, datafile_name =  self._get_dataFileName()
    #     if not os.path.exists(datafile_fullName):
    #         simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
    #     raw_sample_mat = simulativeCloseSeries_df.as_matrix()
    #     train_x_list, train_y_list, valid_x_list, valid_y_list = self.__split_train_valid(raw_sample_mat)
    #     train_x_ndarray, train_x_label_ndarray = np.row_stack(train_x_list), np.hstack(train_y_list)
    #     valid_x_ndarray, valid_x_label_ndarray = np.row_stack(valid_x_list), np.hstack(valid_y_list)
    #
    #     print 'train_x_ndarray', train_x_ndarray.shape
    #     data_parameter_dict = self._data_parameters_obj._args
    #     balancedShuffled_train_x, balancedShuffled_train_y = \
    #         get_balanced_shuffled_datasets(train_x_ndarray, train_x_label_ndarray, data_parameter_dict)
    #     print 'balancedShuffled_train_x.shape', balancedShuffled_train_x.shape
    #     #balancedShuffled_valid_x, balancedShuffled_valid_y = \
    #     #    get_balanced_shuffled_datasets(valid_x_ndarray, valid_x_label_ndarray, data_parameter_dict)
    #     balancedShuffled_valid_x = valid_x_ndarray.reshape(-1, valid_x_ndarray.shape[1], 1, 1)
    #     balancedShuffled_valid_y = valid_x_label_ndarray
    #     #balancedShuffled_valid_x, balancedShuffled_valid_y = valid_x_ndarray, valid_x_label_ndarray
    #     print 'balancedShuffled_valid_x.shape,', balancedShuffled_valid_x.shape
    #     self._train_x, self._train_y = balancedShuffled_train_x, balancedShuffled_train_y
    #     self._valid_x, self._valid_y = balancedShuffled_valid_x, balancedShuffled_valid_y


class Your_Model(base.Model):
    '''
        write your own model
    '''
    def __init__(self, model_parameters_obj):
        self._model_parameters_obj = model_parameters_obj
        self._data_obj = model_parameters_obj._args['data_obj']
        del model_parameters_obj._args['data_obj']
    def _run(self):
        print('###################### model is running ######################')
        self._train()
        evaluate_returns = self._evaluate() #train_confuse_df, valid_confuse_df
        test_returns = self._test()
        ####  save necessary results to file
        self.__write_parameter_to_file()
        ### delete the variables that do not use in the next round
        self._clean()
        return evaluate_returns, test_returns
    def _get_modelInput_from_generalInput(self,x,y):

        nb_classes = self._model_parameters_obj._args['nb_classes']
        y_categorical = to_categorical(y, num_classes=nb_classes)
        x_list, y_categorical_list = self.__from_x_to_xList(x, y_categorical)
        return x_list, y_categorical_list

    def __get_modelFileName(self):

        data_obj = self._data_obj
        parameter_dict = data_obj._data_parameters_obj._args
        parameter_dict['dataType'] = 'train'
        modelFile_name = tool.get_onlyFileName(parameter_dict)

        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        abslute_modelFile_directory = underlyingTime_directory + self._model_parameters_obj._args['modelFile_directoryName']
        tool.make_directory(abslute_modelFile_directory)

        postfix = self._model_parameters_obj._args['modelFile_postfix']
        modelFile_FullName = abslute_modelFile_directory + modelFile_name + postfix
        return modelFile_FullName, modelFile_name

    def __write_parameter_to_file(self):
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameters_obj._args
        model_parameter_dict = self._model_parameters_obj._args
        parameter_directory = tool.get_underlyingTime_directory(data_parameter_dict)
        parameter_fileName = parameter_directory + 'parameter.json'
        file_obj = open(parameter_fileName, 'w')
        save_parameter_dict = [data_parameter_dict, model_parameter_dict]
        file_obj.write(json.dumps(save_parameter_dict))
        file_obj.close()
        # post process parameter data in the file easy to read for people
        file_obj = open(parameter_fileName, 'w')



    def _train(self):
        data_obj = self._data_obj
        self._train_x, self._train_y = data_obj._train_x, data_obj._train_y
        self._valid_x, self._valid_y =data_obj._valid_x, data_obj._valid_y
        self._modelFile_fullName, modelFile_name = self.__get_modelFileName()
        parameter_dict = self._model_parameters_obj._args
        loss, acc, confusion_mat = im.train(self._train_x, self._train_y, self._valid_x, self._valid_y,
                                            self._modelFile_fullName, parameter_dict)
        #self.__train()

    def _evaluate(self):
        #train_confuse_df, valid_confuse_df = self.__evaluate()
        model_parameter_dict = self._model_parameters_obj._args
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameters_obj._args
        predict_train_y_int, predict_valid_y_int = im.evaluate(self._train_x, self._train_y, self._valid_x, self._valid_y,
                                           self._modelFile_fullName, model_parameter_dict)

        y_true = np.array(self._train_y)
        y_predict = np.array(predict_train_y_int)
        data_parameter_dict['dataType'] = 'train'
        train_confuse_df = tool.confuse_matrix(y_true, y_predict, data_parameter_dict)
        y_true = np.array(self._valid_y)
        y_predict = np.array(predict_valid_y_int)
        data_parameter_dict['dataType'] = 'valid'
        valid_confuse_df = tool.confuse_matrix(y_true, y_predict, data_parameter_dict)
        return [train_confuse_df, valid_confuse_df]

    def _test(self):
        model_parameter_dict = self._model_parameters_obj._args
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameters_obj._args
        data_obj._prepare_test_data()
        self._test_x, self._test_y = data_obj._test_x, data_obj._test_y
        im.test(self._test_x, self._test_y, self._modelFile_fullName, model_parameter_dict)
        predict_test_y_int = im.predict(self._test_x, self._modelFile_fullName)
        y_true = np.array(self._test_y)
        y_predict = np.array(predict_test_y_int)
        data_parameter_dict['dataType'] = 'test'
        test_confuse_df = tool.confuse_matrix(y_true, y_predict, data_parameter_dict)
        return [test_confuse_df, y_true, y_predict, data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, data_obj._test_x_date]


    def _clean(self):
        data_obj = self._data_obj
        parameter_dict = data_obj._data_parameters_obj._args
        #data_parameter_dict.pop('currenttime_str') ## internal variable, only used during iteration
        pass


