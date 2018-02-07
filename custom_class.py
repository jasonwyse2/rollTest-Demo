#coding:utf-8
import base_class as base
import numpy as np
import pandas as pd
import os
from data.load_data import get_x_y_sharpGentleUpDown, get_x_y_bottomTopUpDown, get_x_y, get_x_y_repeat
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
    def __init__(self, data_parameter_obj):
        self._data_parameter_obj = data_parameter_obj
        self.__checkLegalInput()
#        self.__dayOrMinute()
    def __checkLegalInput(self):
        #### check legality for train data ###
        train_endTime = self._data_parameter_obj._args['train_endTime']
        if train_endTime=='':
            raise Exception('"train_endTime" can not be empty')
        test_endTime = self._data_parameter_obj._args['test_endTime']
        if test_endTime == '':
            raise Exception('"test_endTime" can not be empty')

    def _get_dataFileName(self):
        parameter_dict = self._data_parameter_obj._args
        file_name = tool.get_onlyFileName(parameter_dict)

        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        abslute_dataFile_directory = underlyingTime_directory + parameter_dict['dataFile_directoryName']
        tool.make_directory(abslute_dataFile_directory)

        postfix = parameter_dict['dataFile_postfix']
        file_FullName = abslute_dataFile_directory + file_name + postfix
        return file_FullName, file_name

    def _run(self):
        self._prepare_trainValid_data_separate()


    def __read_mat_from_csv(self,dataType=''):
        data_parameter_dict = self._data_parameter_obj._args
        csv_path = data_parameter_dict['alpha_csv_path']
        extraTradeDays_afterEndTime = data_parameter_dict['extraTradeDays_afterEndTime']
        NTradeDays_for_indicatorCalculation = data_parameter_dict['NTradeDays_for_indicatorCalculation']
        data_parameter_dict['dataType'] = dataType
        startTime, endTime = tool.get_start_end(data_parameter_dict)
        raw_df = pd.read_csv(csv_path)
        date_list = np.array(raw_df.iloc[:, 0].tolist()).astype(np.str)
        #print('%s date_list'%dataType, date_list[:5])
        start_idx = np.where(date_list >= startTime)[0][0]
        end_idx = np.where(date_list < endTime)[0][-1]
        if start_idx < extraTradeDays_afterEndTime:
            raise Exception('there is no enough data befor %s start_time %s for indicator!' % (dataType,startTime))
        if end_idx + extraTradeDays_afterEndTime > raw_df.shape[0]:
            raise Exception(
                '%s end_time %s must be earlyer than %s' % (dataType, endTime, raw_df.ix[raw_df.shape[0] - 3, 0]))
        if dataType == 'train' or dataType == 'valid':
            raw_sample = raw_df.iloc[:, 1:]
        elif dataType == 'test': # the 0-th column is the time field, we need it to draw figures
            raw_sample = raw_df
        raw_sample_mat = raw_sample.iloc[
                         start_idx-NTradeDays_for_indicatorCalculation :end_idx+1+extraTradeDays_afterEndTime, :]

        return raw_sample_mat

    def _prepare_data(self,dataType=''):
        data_parameter_dict = self._data_parameter_obj._args
        data_parameter_dict['dataType'] = dataType
        dayOrMinute = data_parameter_dict['dayOrMinute']
        minuteType_dict = data_parameter_dict['minuteType_dict']
        if dayOrMinute == 'alpha':
            raw_data_df = self.__read_mat_from_csv(dataType)
            dataToSave_df = raw_data_df
            raw_data_mat = raw_data_df.as_matrix()
            [x_list, y_list, filtered_close_list, close_list] = get_x_y_repeat(raw_data_mat, data_parameter_dict)
        elif dayOrMinute == 'day' or dayOrMinute == minuteType_dict['minuteSimulative']:
            raw_data_df = tool.get_daily_data(data_parameter_dict)
            simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, data_parameter_dict)
            dataToSave_df = simulativeCloseSeries_df

            simulative_sample_matrix = simulativeCloseSeries_df.as_matrix()
            [x_list, y_list, filtered_close_list, close_list] = get_x_y_repeat(simulative_sample_matrix, data_parameter_dict)
        elif dayOrMinute == minuteType_dict['minuteNoSimulative']: # it has no simulative data
            raw_data_df = tool.get_daily_data(data_parameter_dict)
            dataToSave_df = raw_data_df

            x_list, y_list = [], []
            filtered_close_list, close_list = [], []
            x, y, filtered_close_for_use, close_for_use = get_x_y(raw_data_df, data_parameter_dict)
            x_list.append(x)
            y_list.append(y)
            filtered_close_list.append(filtered_close_for_use)
            close_list.append(close_for_use)
        ### save training data ########
        datafile_fullName, datafile_name = self._get_dataFileName()
        if not os.path.exists(datafile_fullName):
            dataToSave_df.to_csv(datafile_fullName, encoding='utf-8')

        return x_list, y_list, filtered_close_list, close_list


    def _prepare_test_data(self):
        data_parameter_dict = self._data_parameter_obj._args
        data_parameter_dict['dataType'] = 'test'
        extra_front = data_parameter_dict['NTradeDays_for_indicatorCalculation']
        extra_end = data_parameter_dict['extraTradeDays_afterEndTime']
        dayOrMinute = data_parameter_dict['dayOrMinute']
        time_field = data_parameter_dict['time_field']
        minuteType_dict = data_parameter_dict['minuteType_dict']
        if dayOrMinute == 'day':
            df = tool.get_daily_data(data_parameter_dict)
            raw_data_df = pd.DataFrame(df['close'])
            df_time_field = df.loc[:, [time_field]]
            self._test_x_date = df_time_field[extra_front:-extra_end]
        elif dayOrMinute == minuteType_dict['minuteNoSimulative']:
            df = tool.get_daily_data(data_parameter_dict)
            raw_data_df = df # we need all field to calculate indicator
            df_time_field = df[time_field]
            self._test_x_date = df_time_field[extra_front:-extra_end]
        elif dayOrMinute == 'alpha':
            df = self.__read_mat_from_csv(dataType='test')
            self._test_x_date = np.array(df.iloc[extra_front:-extra_end,0].tolist()).astype(np.str)
            raw_data_df = pd.DataFrame(df.iloc[:,1])
        elif dayOrMinute == minuteType_dict['minuteSimulative']:
            df = tool.get_daily_data(data_parameter_dict)
            raw_data_df = pd.DataFrame(df['close'])
            df_time_field = df[time_field]
            self._test_x_date = df_time_field[extra_front:-extra_end]
        [self._test_x, self._test_y, self._test_filtered_close_for_use, self._test_close_for_use] = get_x_y(raw_data_df, data_parameter_dict)
        self._test_x = self._test_x.reshape(-1, self._test_x.shape[1], 1, 1)

    def _prepare_trainValid_data_separate(self):
        train_x_list, train_y_list, train_filtered_close_list, train_close_list = self._prepare_data(dataType = 'train')
        valid_x_list, valid_y_list, valid_filtered_close_list, valid_close_list = self._prepare_data(dataType = 'valid')
        train_x_ndarray, train_y_ndarray = np.row_stack(train_x_list), np.hstack(train_y_list)
        valid_x_ndarray, valid_y_ndarray = np.row_stack(valid_x_list), np.hstack(valid_y_list)

        print 'train_x_ndarray', train_x_ndarray.shape
        parameter_dict = self._data_parameter_obj._args
        balancedShuffled_train_x, balancedShuffled_train_y = \
            get_balanced_shuffled_datasets(train_x_ndarray, train_y_ndarray, parameter_dict)
        print 'balancedShuffled_train_x.shape', balancedShuffled_train_x.shape

        balancedShuffled_valid_x = valid_x_ndarray.reshape(-1, valid_x_ndarray.shape[1], 1, 1)
        balancedShuffled_valid_y = valid_y_ndarray
        print 'balancedShuffled_valid_x.shape,', balancedShuffled_valid_x.shape

        self._train_x, self._train_y = balancedShuffled_train_x, balancedShuffled_train_y
        self._valid_x, self._valid_y = balancedShuffled_valid_x, balancedShuffled_valid_y

class Your_Model(base.Model):
    '''
        write your own model
    '''
    def __init__(self, model_parameter_obj):
        self._model_parameter_obj = model_parameter_obj
        self._data_obj = model_parameter_obj._args['data_obj']
        del model_parameter_obj._args['data_obj']
    def _run(self):
        print('###################### model is running ######################')
        self._train()
        evaluate_returns = self._evaluate() #train_confuse_df, valid_confuse_df
        test_returns = self._test()
        ####  save necessary results to file
        self.__write_parameter_to_file()
        ### delete the variables that do not use in the next round
#        self._clean()
        return evaluate_returns, test_returns


    def __get_modelFileName(self):

        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameter_obj._args
        data_parameter_dict['dataType'] = 'train'
        data_parameter_dict['isModelName'] = True
        modelFile_name = tool.get_onlyFileName(data_parameter_dict)
        del data_parameter_dict['isModelName']
        underlyingTime_directory = tool.get_underlyingTime_directory(data_parameter_dict)
        abslute_modelFile_directory = underlyingTime_directory + self._model_parameter_obj._args['modelFile_directoryName']
        tool.make_directory(abslute_modelFile_directory)

        postfix = self._model_parameter_obj._args['modelFile_postfix']
        modelFile_FullName = abslute_modelFile_directory + modelFile_name + postfix
        return modelFile_FullName, modelFile_name

    def __write_parameter_to_file(self):
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameter_obj._args
        model_parameter_dict = self._model_parameter_obj._args
        parameter_directory = tool.get_underlyingTime_directory(data_parameter_dict)
        parameter_fileName = parameter_directory + 'parameter.json'
        file_obj = open(parameter_fileName, 'w')
        save_parameter_dict = {'data_parameters_dict':data_parameter_dict, 'model_parameter_dict':model_parameter_dict}
        file_obj.write(json.dumps(save_parameter_dict))
        file_obj.close()
        # post process parameter data in the file easy to read for people
        file_read_obj = open(parameter_fileName, 'r')
        parameter_file_write = parameter_directory + 'easyToRead_parameter.txt'
        file_write_obj = open(parameter_file_write, 'w')
        line = file_read_obj.readline()
        items = line.split(',')
        for item in items:
            file_write_obj.write(item)
            file_write_obj.write('\n')

    def _train(self):
        data_obj = self._data_obj
        self._train_x, self._train_y = data_obj._train_x, data_obj._train_y
        self._valid_x, self._valid_y =data_obj._valid_x, data_obj._valid_y
        self._modelFile_fullName, self._modelFile_name = self.__get_modelFileName()

        model_parameter_dict = self._model_parameter_obj._args
        loss, acc, confusion_mat = im.train(self._train_x, self._train_y, self._valid_x, self._valid_y,
                                            self._modelFile_fullName, model_parameter_dict)

    def _evaluate(self):
        model_parameter_dict = self._model_parameter_obj._args
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameter_obj._args
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
        model_parameter_dict = self._model_parameter_obj._args
        data_obj = self._data_obj
        data_parameter_dict = data_obj._data_parameter_obj._args
        data_obj._prepare_test_data()

        im.test(data_obj._test_x, data_obj._test_y, self._modelFile_fullName, model_parameter_dict)
        predict_test_y_int = im.predict(data_obj._test_x, self._modelFile_fullName)
        y_true = np.array(data_obj._test_y)
        y_predict = np.array(predict_test_y_int)
        data_parameter_dict['dataType'] = 'test'
        test_confuse_df = tool.confuse_matrix(y_true, y_predict, data_parameter_dict)
        return [test_confuse_df, y_true, y_predict, data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, data_obj._test_x_date]
