#coding:utf-8
import base_class as base
import numpy as np
import pandas as pd
import os
from data.datalib.slide_Htargets_datasets import get_datasets_2
from data.datalib.slide_Htargets_datasets import get_balanced_shuffled_datasets
from keras.utils.np_utils import to_categorical
from keras.layers import Input, BatchNormalization,Dropout,Convolution2D, Dense,Flatten,merge
from keras.regularizers import l2
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.models import Sequential,load_model,Model
import tool as tool
import time

import keras


def __get_tenserflow_inputOutput(self):
    train_num = self._model_parameters_obj._args['train_num']
    batch_size = self._model_parameters_obj._args['batch_size']
    nb_classes = self._model_parameters_obj._args['nb_classes']
    save_name = 'a.h5'
    parameter_dict = self._model_parameters_obj._args['data_obj']._data_parameters_obj._args
    data_obj = self._model_parameters_obj._args['data_obj']
    train_x, train_y = data_obj._train_x, data_obj._train_y
    valid_x, valid_y = data_obj._valid_x, data_obj._valid_y

    data_fit_example_shape = train_x.shape
    train_labels = to_categorical(train_y, num_classes=nb_classes)  # ***
    valid_labels = to_categorical(valid_y, num_classes=nb_classes)  # ***
    Input_shape = (1, data_fit_example_shape[2], data_fit_example_shape[3])
    nb_input = data_fit_example_shape[1]
    nb_layer = nb_input - 1

    cov_list, input_list, output_list = [], [], []
    for i in range(nb_input):
        input = Input(shape=Input_shape)
        input_list.append(input)
        noise = BatchNormalization(axis=1)(input)
        cov = self.__CNN_block(noise, 1, 1, 16)
        cov_1 = self.__CNN_block(cov, 1, 1, 16)
        x_out = Flatten()(cov)
        cov_list.append(cov_1)
        out = Dense(nb_classes, activation='softmax')(x_out)
        output_list.append(out)
    out = merge(cov_list, mode='concat', concat_axis=1)
    for j in range(nb_layer):
        out = self.__CNN_block(out, 2, 1, 16)
    out_flat = Flatten()(out)
    out = Dense(nb_classes, activation='softmax')(out_flat)
    output_list.append(out)

    return input_list, output_list

def evaluate(train_x,train_y,valid_x,vliad_y,save_name,parameter_dict):
    #train_int, valid_int =im.evaluate(train_x,train_y,valid_x,vliad_y,save_name,data_parameters_dict)
    #return train_int,valid_int
    pass

def test(test_x,test_y,save_name,parameter_dict):
    #confu_df = im.test(test_x,test_y,save_name,data_parameters_dict)
    #return confu_df
    pass

def predict(x,save_name):
    #pred_int = im.predict(x,save_name)
    #return pred_int
    pass

def show_confusion(y_true,y_pred,nb_classes):
    #confu_df = im.show_confusion(y_true,y_pred,nb_classes)
    #return confu_df
    pass

if __name__ == '__main__':
    time1 = time.time()
    for i in range(1):
        time.sleep(2)
    time2 = time.time()
    print('elapse time:',time2-time1)