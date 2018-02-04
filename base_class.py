#coding:utf-8
import os
import sys
import pandas as pd
import pymysql
from abc import ABCMeta, abstractmethod
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
class Data:
    """
    Base class for storing data.
        Attributes:
            name: it indicates what these data are used for, default:'data'
            args: it's a varying length dictionary
    """
    def __init__(self, name='data', **args):
        self._name = name
        self._args = args

        # default database information,
        self._host = '192.168.1.11'
        self._port = 3306
        self._user = 'zqfordinary'
        self._passwd = 'Ab123456'
        self._db = 'stock'

    def prepare_data(self):
        print('prepare data...')
    def read_data_from_file(self,file_path):
        '''
        :param parameter: pass a dictionary dataType variable
        :return: return a dataframe dataType retult
        '''
        #file_path = parameter['file_path']
        #csv_path = '/mnt/aidata/生成数据/加噪数据/000905generate_dataset02.csv'
        raw_df = pd.read_csv(file_path)
        return raw_df
        print('read_data_from_file')
    def read_data_from_database(self,parameter):
        '''
        :param parameter: pass a dictionary dataType variable
        :return: return a dataframe dataType retult
        '''
        connect = pymysql.Connect(self._host, self._port, self._portuser, self._passwd, self._db)
        print('read_data_from_file')
class Model:
    """
        Base class for model, specific model should inherit this base model. All methods in
        this class can be overloaded by subclass

            Attributes:
                model_standard_input_shape: it indicates the standard input shape

        """
    def __init__(self,model_standard_input_shape=None, data_dict = None,**args):
        # if model_standard_input_shape == None:
        #     raise ValueError('you need to specify the standard input shape for the model')
        # else:
        #     self._model_standard_input_shape = model_standard_input_shape
        self._args = args
        self._data_dict = data_dict

    @abstractmethod
    def check_input_shape(self,input_shape):
        if self._model_standard_input_shape != input_shape:
            raise ValueError('input shape is not equal with the specified model input shape')
        else:
            print('shape same')
    def save_model(self, save_directory, file_name, model):
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)


    def load_model(file_path, custom_objects=None, compile=True):
            print('loading model')

    def run(self):
        print('model running')

    @abstractmethod
    def _train(self):
        print('trainging')
    def _predict(self):
        print('predict')
    def _evaluate(self):
        print('evaluate')

class Test:
    '''

    '''
    def __init__(self,**args):
        self._args = args
    @abstractmethod
    def _test(self, model, test_x = None, test_y =None):
        pass
class Parameters:
    """
    Basic class for parameter transfer. all parameters must be passing by as a dictionary.
    Storing all parameters that given by the class construction function
    Attributes:
        name: it indicates what these parameters are used for,
        args: it's a varying length dictionary
    """
    def __init__(self,name = 'parameters',**args):
        self._name = name
        self._args = args

if __name__ == '__main__':
    data = Data()
    data_parameter = {file_path}

    data.read_data_from_file()
    print(data._args)