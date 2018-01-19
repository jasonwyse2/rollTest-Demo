#coding:utf-8
import base_class as base
import numpy as np
import pandas as pd
import os
from data.load_data import get_datasets_2
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
        #self._data_dict = self._prepare_data()
        self._prepare_trainValid_data()

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

    def __split_train_valid(self, raw_sample_mat):
        '''
        it contains cutting the extra days for tagging data
        :param raw_sample_mat:
        :param lookBack_from_validEndTime:
        :return:
        '''
        valid_lookBack_from_endTime = self._data_parameters_obj._args['valid_lookBack_from_endTime']

        extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']

        train_x_list, train_y_list = [], []
        valid_x_list, valid_y_list = [], []
        closeSeries_num = raw_sample_mat.shape[1]
        show_label = self._data_parameters_obj._args['show_label']
        for i in range(closeSeries_num):
            close_array = raw_sample_mat[:, i]
            # pct = np.diff(close_array) / close_array[:-1]
            # close_array = close_array[1:]
            close = pd.Series(close_array)
            x, y, filtered_close_for_use, close_for_use = get_datasets_2(close, self._data_parameters_obj._args,show_label=show_label)

            train_x, train_y = x[:-(valid_lookBack_from_endTime+extraTradeDays)], y[:-(valid_lookBack_from_endTime+extraTradeDays)]
            valid_x, valid_y = x[-(valid_lookBack_from_endTime+extraTradeDays):-extraTradeDays],\
                               y[-(valid_lookBack_from_endTime+extraTradeDays):-extraTradeDays]
            train_x_list.append(train_x)
            train_y_list.append(train_y)
            valid_x_list.append(valid_x)
            valid_y_list.append(valid_y)
        return train_x_list, train_y_list, valid_x_list, valid_y_list

    def _prepare_trainValid_data(self):
        self._data_parameters_obj._args['dataType'] = 'train_valid'
        raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)


        simulativeCloseSeries_num = self._data_parameters_obj._args['simulativeCloseSeries_num']
        # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
        simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df.index, raw_data_df.close, simulativeCloseSeries_num)
        datafile_fullName, datafile_name =  self._get_dataFileName()
         #simulativeCloseSeries_fileName = \
        #    tool.get_fileName_by_startEndLookback(self._data_parameters_obj._args, dataType='train_valid')
        if not os.path.exists(datafile_fullName):
            simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
        raw_sample_mat = simulativeCloseSeries_df.as_matrix()
        train_x_list, train_y_list, valid_x_list, valid_y_list = self.__split_train_valid(raw_sample_mat)
        train_x_ndarray, train_x_label_ndarray = np.row_stack(train_x_list), np.hstack(train_y_list)
        valid_x_ndarray, valid_x_label_ndarray = np.row_stack(valid_x_list), np.hstack(valid_y_list)

        print 'train_x_ndarray', train_x_ndarray.shape
        parameter_dict = self._data_parameters_obj._args
        balancedShuffled_train_x, balancedShuffled_train_y = \
            get_balanced_shuffled_datasets(train_x_ndarray, train_x_label_ndarray, parameter_dict)
        print 'balancedShuffled_train_x.shape', balancedShuffled_train_x.shape
        #balancedShuffled_valid_x, balancedShuffled_valid_y = \
        #    get_balanced_shuffled_datasets(valid_x_ndarray, valid_x_label_ndarray, parameter_dict)
        balancedShuffled_valid_x = valid_x_ndarray.reshape(-1, valid_x_ndarray.shape[1], 1, 1)
        balancedShuffled_valid_y = valid_x_label_ndarray
        #balancedShuffled_valid_x, balancedShuffled_valid_y = valid_x_ndarray, valid_x_label_ndarray
        print 'balancedShuffled_valid_x.shape,', balancedShuffled_valid_x.shape
        self._train_x, self._train_y = balancedShuffled_train_x, balancedShuffled_train_y
        self._valid_x, self._valid_y = balancedShuffled_valid_x, balancedShuffled_valid_y

    def _prepare_test_data(self):
        self._data_parameters_obj._args['dataType'] = 'test'
        df = tool.get_daily_data(self._data_parameters_obj._args)
        close_array = np.array(df['close'].tolist())
        close = pd.Series(close_array)
        # pct = np.diff(close_array) / close_array[:-1]
        self._test_x, self._test_y, self._test_filtered_close_for_use, self._test_close_for_use = get_datasets_2(close,self._data_parameters_obj._args,show_label=False)
        ### cut extra days for tagging data, in order to keep consistent with date given in 'data_parameters_obj'
        extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
        self._test_x, self._test_y,  = self._test_x[:-extraTradeDays], self._test_y[:-extraTradeDays]
        self._test_filtered_close_for_use, self._test_close_for_use = self._test_filtered_close_for_use[:-extraTradeDays], self._test_close_for_use[:-extraTradeDays]
        self._test_x = self._test_x.reshape(-1, self._test_x.shape[1], 1, 1)


class Your_Model(base.Model):
    '''
        write your own model
    '''
    def __init__(self, model_parameters_obj):
        self._model_parameters_obj = model_parameters_obj
    def _run(self):
        print('###################### model is running ######################')
        self._train()
        self._evaluate()
        self._test()

        ### delete the variables that do not use in the next round
        self._clean()


    def __get_modelFileName(self):

        data_obj = self._model_parameters_obj._args['data_obj']
        parameter_dict = data_obj._data_parameters_obj._args
        modelFile_name = tool.get_onlyFileName(parameter_dict)

        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        abslute_modelFile_directory = underlyingTime_directory + self._model_parameters_obj._args['modelFile_directoryName']
        tool.make_directory(abslute_modelFile_directory)

        postfix = self._model_parameters_obj._args['modelFile_postfix']
        modelFile_FullName = abslute_modelFile_directory + modelFile_name + postfix
        return modelFile_FullName, modelFile_name

    def __write_parameter_to_file(self):
        data_obj = self._model_parameters_obj._args['data_obj']
        data_parameter_dict = data_obj._data_parameters_obj._args
        model_parameter_dict = self._model_parameters_obj._args
        parameter_directory = tool.get_underlyingTime_directory(data_parameter_dict)
        parameter_fileName = parameter_directory + 'parameter.json'
        file_obj = open(parameter_fileName, 'w')
        save_parameter_dict = {'train_startTime':data_parameter_dict['train_startTime']}
        file_obj.write(json.dumps(save_parameter_dict))

    def _train(self):
        data_obj = self._model_parameters_obj._args['data_obj']
        self._train_x, self._train_y = data_obj._train_x, data_obj._train_y
        self._valid_x, self._valid_y =data_obj._valid_x, data_obj._valid_y

        modelFile_fullName, modelFile_name = self.__get_modelFileName()
        train_num = self._model_parameters_obj._args['train_num']
        batch_size = self._model_parameters_obj._args['batch_size']
        #verbose=1: dynamicly show the progress, verbose=0: show nothing
        im.train(self._train_x,self._train_y,self._valid_x,self._valid_y,train_num,batch_size,modelFile_fullName)
        self._model = im

    def _load_model_from_file(self):
        # modelFile_fullName = self.__get_modelFileName()
        # self._model = load_trained_model(modelFile_fullName)
        pass

    def _loadModel_evaluate(self):
        # modelFileName = self.__get_modelFileName()
        # if os.path.exists(modelFileName):
        #     self._model =load_trained_model(modelFileName)
        # self._evaluate()
        pass

    def _evaluate(self):
        # evaluate_batch_size = self._model_parameters_obj._args['evaluate_batch_size']
        # evaluate_verbose = self._model_parameters_obj._args['evaluate_verbose']
        # model = self._model
        data_obj = self._model_parameters_obj._args['data_obj']
        self._train_x, self._train_y = data_obj._train_x, data_obj._train_y
        self._valid_x, self._valid_y = data_obj._valid_x, data_obj._valid_y

        modelFile_fullName, modelFile_name = self.__get_modelFileName()
        im.evaluate(self._train_x,self._train_y,self._valid_x,self._valid_y,modelFile_fullName,self._model_parameters_obj._args)

    def _test(self):

        data_obj = self._model_parameters_obj._args['data_obj']

        data_obj._prepare_test_data()
        self._test_x,self._test_y = data_obj._test_x, data_obj._test_y
#        self._model_parameters_obj._args['data_obj']._data_parameters_obj._args['dataType'] = 'train_valid'
        data_obj._data_parameters_obj._args['dataType'] = 'train_valid'
        modelFile_fullName, modelFile_name = self.__get_modelFileName()
        print 'save_name',modelFile_fullName
        predict_test_y_toInt = im.predict(self._test_x,modelFile_fullName)

        y_true = np.array(data_obj._test_y)
        y_predict = np.array(predict_test_y_toInt)
        print('****test confusion matrix****')
        confuse_df = tool.show_confuse_matrix(y_true, y_predict)
        confu = confuse_df.as_matrix()
        pred_acc_per_class = np.diag(confu).astype(np.float) / np.sum(confu, axis=0)
        print 'predict_accuracy_per_class:'
        print pred_acc_per_class
        print 'Average accuracy:', np.mean(pred_acc_per_class)

        parameter_dict = data_obj._data_parameters_obj._args
        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        dataType = parameter_dict['dataType']
        parameter_dict['dataType'] = 'test'

        test_onlyFileName = tool.get_onlyFileName(parameter_dict)
        abslute_test_filename = underlyingTime_directory + test_onlyFileName+'.csv'
        confuse_df.to_csv(abslute_test_filename)

        ###  save figure results ###
        startTime, endTime, train_lookBack, valid_lookBack = tool.get_start_end_lookback(parameter_dict)
        true_tag_figure_path = underlyingTime_directory + '%s-%s-TrueLabels.png' % (startTime, endTime)
        tool.save_fig(data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, y_true, true_tag_figure_path)
        predict_tag_figure_path = underlyingTime_directory + '%s-%s-PredictLabels.png' % (startTime, endTime)
        tool.save_fig(data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, y_predict, predict_tag_figure_path)



    def _clean(self):
        data_obj = self._model_parameters_obj._args['data_obj']
        parameter_dict = data_obj._data_parameters_obj._args
        #parameter_dict.pop('currenttime_str') ## internal variable, only used during iteration
        self.__write_parameter_to_file()
        pass


