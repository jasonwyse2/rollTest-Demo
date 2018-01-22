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
        raw_data_df = tool.get_daily_data(self._data_parameters_obj._args)
        # each column in 'simulativeCloseSeries_df' is a simulative close series, which can be generated by different generator
        simulativeCloseSeries_df = tool.simulative_close_generator(raw_data_df, self._data_parameters_obj._args)
        datafile_fullName, datafile_name = self._get_dataFileName()
        if not os.path.exists(datafile_fullName):
            simulativeCloseSeries_df.to_csv(datafile_fullName, encoding='utf-8')
        simulative_sample_matrix = simulativeCloseSeries_df.as_matrix()

        closeSeries_num = simulative_sample_matrix.shape[1]
        x_list, y_list = [], []
        extraTradeDays = self._data_parameters_obj._args['extraTradeDays_afterEndTime_for_filter']
        for i in range(closeSeries_num):
            close_array = simulative_sample_matrix[:, i]
            close = pd.Series(close_array)
            x, y, filtered_close_for_use, close_for_use = get_x_y(close, self._data_parameters_obj._args)
            ## show the figure
            # if dataType == 'valid':
            #     tool.show_fig(y, filtered_close_for_use, close_for_use)
            #train_x, train_y = x[:-(extraTradeDays)], y[:-(extraTradeDays)]
            x_list.append(x)
            y_list.append(y)
        return x_list, y_list


    def _prepare_test_data(self):
        self._data_parameters_obj._args['dataType'] = 'test'
        df = tool.get_daily_data(self._data_parameters_obj._args)
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
        #    get_balanced_shuffled_datasets(valid_x_ndarray, valid_x_label_ndarray, parameter_dict)
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
    #     parameter_dict = self._data_parameters_obj._args
    #     balancedShuffled_train_x, balancedShuffled_train_y = \
    #         get_balanced_shuffled_datasets(train_x_ndarray, train_x_label_ndarray, parameter_dict)
    #     print 'balancedShuffled_train_x.shape', balancedShuffled_train_x.shape
    #     #balancedShuffled_valid_x, balancedShuffled_valid_y = \
    #     #    get_balanced_shuffled_datasets(valid_x_ndarray, valid_x_label_ndarray, parameter_dict)
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
    def _run(self):
        print('###################### model is running ######################')
        self._train()
        train_confu_matrix_df, valid_confu_matrix_df = self._evaluate()
        test_confu_matrix_df = self._test()
        ####  save necessary results to file
        self.__write_parameter_to_file()
        ### delete the variables that do not use in the next round
        self._clean()
        return train_confu_matrix_df, valid_confu_matrix_df, test_confu_matrix_df
    def _get_modelInput_from_generalInput(self,x,y):

        nb_classes = self._model_parameters_obj._args['nb_classes']
        y_categorical = to_categorical(y, num_classes=nb_classes)
        x_list, y_categorical_list = self.__from_x_to_xList(x, y_categorical)
        return x_list, y_categorical_list

    def __CNN_block(self, out, filter_row, filter_col, nb_filter):
        out_signal_layer = Convolution2D(nb_filter, filter_row, filter_col, init='normal', activation='relu',
                                         border_mode='valid', W_regularizer=l2(0.001), subsample=(1, 1),
                                         dim_ordering='tf', bias=True)  # ,W_constraint=maxnorm(m=1.2),
        out_signal = out_signal_layer(out)
        out_signal = BatchNormalization(axis=3)(out_signal)
        # out_signal = Activation('relu')(out_signal)
        out_signal = Dropout(0.1)(out_signal)
        out_put_shape = out_signal_layer.output_shape
        return out_signal


    def __get_CNN_tenserflow_inputOutput(self):

        data_obj = self._model_parameters_obj._args['data_obj']
        nb_classes = self._model_parameters_obj._args['nb_classes']
        train_x = data_obj._train_x
        data_fit_example_shape = train_x.shape

        '''
        When using  this layer as the first layer in a model, provide the keyword argument `input_shape` (tuple of integers, does
        not include the sample axis), e.g. `input_shape = (128, 128, 3)` for 128x128 RGB pictures in `data_format = "channels_last"
        `.
        '''
        Input_shape = (data_fit_example_shape[1], data_fit_example_shape[2], 1)

        filter_row = self._model_parameters_obj._args['filter_row']
        filter_col = self._model_parameters_obj._args['filter_col']
        filter_num = self._model_parameters_obj._args['filter_num']

        input_list, output_list = [], []
        input = Input(shape=Input_shape)
        input_list.append(input)

        normalize_input = BatchNormalization(axis=3)(input)
        cov = self.__CNN_block(normalize_input, filter_row, filter_col, filter_num)

        cov = self.__CNN_block(cov, filter_row, filter_col, filter_num)
        cov = self.__CNN_block(cov, filter_row, filter_col, filter_num)
        cov = self.__CNN_block(cov, filter_row, filter_col, filter_num)
        cov = self.__CNN_block(cov, filter_row, filter_col, filter_num)

        out_flat = Flatten()(cov)
        out = Dense(nb_classes, activation='softmax')(out_flat)
        output_list.append(out)

        return input_list, output_list

    # def __get_tenserflow_inputOutput(self):
    #
    #     nb_classes = self._model_parameters_obj._args['nb_classes']
    #     data_obj = self._model_parameters_obj._args['data_obj']
    #     train_x, train_y = data_obj._train_x, data_obj._train_y
    #     valid_x, valid_y = data_obj._valid_x, data_obj._valid_y
    #     nb_classes = self._model_parameters_obj._args['nb_classes']
    #     train_y_categorical = to_categorical(train_y, num_classes=nb_classes)
    #     valid_y_categorical = to_categorical(valid_y, num_classes=nb_classes)
    #
    #     data_fit_example_shape = train_x.shape
    #
    #     Input_shape = (1, data_fit_example_shape[2], data_fit_example_shape[3])
    #     nb_input = data_fit_example_shape[1]
    #     nb_layer = nb_input - 1
    #
    #     cov_list, input_list, output_list = [], [], []
    #     for i in range(nb_input):
    #         input = Input(shape=Input_shape)
    #         input_list.append(input)
    #         noise = BatchNormalization(axis=1)(input)
    #         cov = self.__CNN_block(noise, 1, 1, 16)
    #         cov_1 = self.__CNN_block(cov, 1, 1, 16)
    #         x_out = Flatten()(cov)
    #         cov_list.append(cov_1)
    #         out = Dense(nb_classes, activation='softmax')(x_out)
    #         output_list.append(out)
    #     out = merge(cov_list, mode='concat', concat_axis=1)
    #     for j in range(nb_layer):
    #         out = self.__CNN_block(out, 2, 1, 16)
    #     out_flat = Flatten()(out)
    #     out = Dense(nb_classes, activation='softmax')(out_flat)
    #     output_list.append(out)
    #     return input_list, output_list

    def __train(self):
        # loss,acc, confusion_mat = im.train(train_x,train_y,valid_x,vliad_y,save_name,parameter_dict)
        # return loss ,acc, confusion_mat
        data_obj = self._model_parameters_obj._args['data_obj']
        nb_classes = self._model_parameters_obj._args['nb_classes']
        self._train_y_categorical = to_categorical(data_obj._train_y, num_classes=nb_classes)
        self._valid_y_categorical = to_categorical(data_obj._valid_y, num_classes=nb_classes)
        tenserflowInput_list, tenserflowOutput_list = self.__get_CNN_tenserflow_inputOutput()

        model = Model(input=tenserflowInput_list, output=tenserflowOutput_list)


        rmsprop = RMSprop(lr=0.001, rho=0.9, decay=0.99)

        model.compile(loss='categorical_crossentropy', optimizer=rmsprop,
                      metrics=['accuracy'])  # set loss function and optimizer #loss_weights=loss_weights,

        modelFile_FullName, modelFile_name = self.__get_modelFileName()
        modelcheck = ModelCheckpoint(modelFile_FullName, monitor='val_loss', save_best_only=True)
        tensorboard = TensorBoard(log_dir='tensorboard', histogram_freq=0, write_graph=True)
        train_epochs = self._model_parameters_obj._args['train_epoch']
        batch_size = self._model_parameters_obj._args['batch_size']
        # verbose=1: dynamicly show the progress, verbose=0: show nothing
        model.fit(self._train_x, self._train_y_categorical, batch_size=batch_size, nb_epoch = train_epochs,
                  verbose=1,
                  validation_data=(self._valid_x, self._valid_y_categorical),
                  callbacks=[modelcheck, tensorboard])
        self._model = model

    def __evaluate(self):
        # loss,acc, confusion_mat = im.train(train_x,train_y,valid_x,vliad_y,save_name,parameter_dict)
        # return loss ,acc, confusion_mat

        evaluate_batch_size = self._model_parameters_obj._args['evaluate_batch_size']
        evaluate_verbose = self._model_parameters_obj._args['evaluate_verbose']
        model = self._model
        data_obj = self._model_parameters_obj._args['data_obj']
        idx = data_obj._train_x.shape[1]
        train_result = model.evaluate(self._train_x, self._train_y_categorical,
                                      batch_size=evaluate_batch_size, verbose=evaluate_verbose)
        evaluate_result = model.evaluate(self._valid_x, self._valid_y_categorical,
                                         batch_size=evaluate_batch_size, verbose=evaluate_verbose)

        print('Train loss: %f, Train acc: %f, Valid loss: %f, Valid acc: %f') % (
            train_result[0], train_result[1], evaluate_result[0], evaluate_result[1])

        predict_train_y_categorical = model.predict(self._train_x)
        predict_train_y_toInt = np.argmax(predict_train_y_categorical, axis=1)
        y_true = np.array(data_obj._train_y)
        y_predict = np.array(predict_train_y_toInt)
        print('####### train confusion matrix ########')
        train_confu_matrix_df = tool.show_confuse_matrix(y_true, y_predict)


        predict_valid_y_categorical = model.predict(self._valid_x)
        predict_valid_y_toInt = np.argmax(predict_valid_y_categorical, axis=1)
        y_true = np.array(data_obj._valid_y)
        y_predict = np.array(predict_valid_y_toInt)
        print('####### valid confusion matrix ########')
        valid_confu_matrix_df = tool.show_confuse_matrix(y_true, y_predict)
        return train_confu_matrix_df, valid_confu_matrix_df

    def __test(self):
        data_obj = self._model_parameters_obj._args['data_obj']
        data_obj._prepare_test_data()
        self._test_x, self._test_y = data_obj._test_x, data_obj._test_y
        #        self._model_parameters_obj._args['data_obj']._data_parameters_obj._args['dataType'] = 'train_valid'
        data_obj._data_parameters_obj._args['dataType'] = 'train'
        modelFile_fullName, modelFile_name = self.__get_modelFileName()
        print 'save_name', modelFile_fullName
        predict_test_y_categorical = self._model.predict(self._test_x)
        predict_test_y_toInt = np.argmax(predict_test_y_categorical, axis=1)
        y_true = np.array(data_obj._test_y)
        y_predict = np.array(predict_test_y_toInt)

        print('****test confusion matrix****')
        test_confuse_df = tool.show_confuse_matrix(y_true, y_predict)

        parameter_dict = data_obj._data_parameters_obj._args
        underlyingTime_directory = tool.get_underlyingTime_directory(parameter_dict)
        dataType = parameter_dict['dataType']

        parameter_dict['dataType'] = 'test'
        test_onlyFileName = tool.get_onlyFileName(parameter_dict)
        abslute_test_filename = underlyingTime_directory + test_onlyFileName + '.csv'
        test_confuse_df.to_csv(abslute_test_filename)

        parameter_dict['dataType'] = dataType
        ###  save figure results ###
        startTime, endTime, train_lookBack, valid_lookBack = tool.get_start_end_lookback(parameter_dict)
        true_tag_figure_path = underlyingTime_directory + '%s-%s-TrueLabels.png' % (startTime, endTime)
        tool.save_fig(data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, y_true, true_tag_figure_path)
        predict_tag_figure_path = underlyingTime_directory + '%s-%s-PredictLabels.png' % (startTime, endTime)
        tool.save_fig(data_obj._test_filtered_close_for_use, data_obj._test_close_for_use, y_predict,
                      predict_tag_figure_path)
        return test_confuse_df

    def __get_modelFileName(self):

        data_obj = self._model_parameters_obj._args['data_obj']
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

        #self._model = im
        self.__train()

    def _evaluate(self):
        train_confu_matrix_df, valid_confu_matrix_df = self.__evaluate()
        return train_confu_matrix_df, valid_confu_matrix_df

    def _test(self):
        test_confuse_matrix_df = self.__test()
        return test_confuse_matrix_df

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



    def _clean(self):
        data_obj = self._model_parameters_obj._args['data_obj']
        parameter_dict = data_obj._data_parameters_obj._args
        #parameter_dict.pop('currenttime_str') ## internal variable, only used during iteration
        pass


