#coding:utf-8
# Model Training
# Module or toolbox required list: (1) Tensorflow(2)Keras(3)Python(4)Numpy(5)Pandas(6)talib(8)MySQL python
from __future__ import absolute_import
import argparse
import numpy as np
import pandas as pd

import os
import sys

from engine import index_model as im
from data.datalib.slide_Htargets_datasets import get_datasets_2
from data.datalib.slide_Htargets_datasets import get_balanced_shuffled_datasets
from itertools import permutations
from load_data_from_database import query_kdata
# from sklearn import metric
def model_predict(test_x,save_name):
# data : numpy array, input data, format:28*5*n(n is the number of samples)
# label : numpy array , input labels, for n*1 vector
# var_select : bool, whether to activate the input variables selecting procedure
# selected_var : list, manually select the which input varibales are used for model training
# train_num : int, number of steps for training
# train_valid_ratio: float in [0,1], the ratio of training data and valid data splited from input data.
# unbalance_ratio: float in [0,inf], suggest between [1,3], the ratio of number of samples in major class over number of samples in minor class.
#                  The system will auto match this ratio when construct the dataset from original
# save_name: str, name of model for saving
# var_select_load: bool, whether to load the index of variables that are selected for training.

# ------------------------------------ Default parameters, for advanced use only ------------------------------
    seed = 585 # random seed

# -------------------------------------------------------------------------------------------------------------
    current_path=os.getcwd()

    model_save_path = current_path+'/models/'+save_name+'.h5'
    pred_int = im.predict(test_x,model_save_path)
    return  pred_int


def get_confusion_matrix(pred_int, real_int):
    confusion_mat = np.zeros(shape=(4, 4))
    possible_tupe = list(permutations(range(4), 2)) + [(i, i) for i in range(4)]
    for v in possible_tupe:
        confusion_mat[v[0], v[1]] = list(zip(pred_int, real_int)).count(v)

    return confusion_mat

if __name__ == '__main__':
    sql = 'SELECT DATE,CLOSE,pct_chg FROM index_date WHERE DATE>20160101 AND DATE<20171231 AND code_id = 3042'
    seed = 4
    raw_df = query_kdata(sql)

    # save_name = '512batch'
    # save_name = '20120101-20150101'
    # save_name = 'batch_rmsprop'

    save_name = '20050121-20161031'
    date_list = np.array(raw_df.iloc[:, 0].tolist())

    #sample_for_test = np.array(raw_df.iloc[start_index - 240 - time_len + 1:last_index + 1 + 1, 2])
    date_for_test = raw_df.iloc[:, 0]
    sample_for_test = raw_df.iloc[:, 2]

    print 'sample_for_test shape', sample_for_test.shape[0]
    print 'date_for_test', date_for_test

    raw_sample = np.array(sample_for_test.astype('double').tolist())/100
    print 'raw_smaple',type(raw_sample)
    close = np.array(raw_df.iloc[:,1].astype('double').tolist())
    print 'close',close[:20]

    test_x, test_y, filtered_close_for_use,close_for_use = get_datasets_2(close, raw_sample)

    print 'test_y',set(test_y),test_y.shape
    print 'sample_for_test shape', sample_for_test.shape[0]
    print 'date_for_test', date_for_test

    print 'test_x.shape', test_x.shape
    test_x = test_x.transpose(0,2,1).reshape(-1,12,20,1)
    pred_int = model_predict(test_x, save_name=save_name)
    print 'pred_int',set(pred_int),pred_int.shape

    real_int = test_y
    # print pred_int[:10]
    # print real_int[:10]

    print len(pred_int)
    print len(np.where((np.array(pred_int) - np.array(real_int)) == 0)[0])
    print 'test acc', float(len(np.where((np.array(pred_int) - np.array(real_int)) == 0)[0])) / len(pred_int)
    print('### Model training complete! ###')

   # raw_close_rate = np.cumsum(raw_log_return)
    #raw_close = np.exp(raw_close_rate)
    # mean,std = np.mean(raw_sample[:240]),np.std(raw_sample[:240])
    # print('mean: %f , std: %f')%(mean,std)
    #result_dict = {'predict_int': pred_int, 'pct': raw_log_return[240 + time_len - 1:-1],
  #                 'raw_close': raw_close[240 + time_len - 1:-1], 'date': date_for_test.tolist()[240 + time_len - 1:-1],
    #               'real_int': real_int}
    #df = pd.DataFrame(result_dict)
    #df.index = range(pred_int.shape[0])
   # df.to_csv('predict_states_' + save_name[-8:] + '.csv')

    confusion_mat = get_confusion_matrix(pred_int, real_int)
    df = pd.DataFrame(confusion_mat)
    confusion_csv = 'confusion' + save_name[-8:] + '.csv'
    #df.to_csv(confusion_csv)
    print 'test_confusion'

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(range(len(close_for_use[19:])), close_for_use[19:], c='r')
    plt.plot(range(len(close_for_use[19:])), filtered_close_for_use[19:], c='b')
    # print np.where(labels==1)[0]
    # print np.where(labels == 2)[0]
    # print np.where(labels == 3)[0]
    # print np.where(labels == 4)[0]
    # print type(filtered_close_for_use)
    import matplotlib.pyplot as plt

    filtered_close_for_use = np.array(filtered_close_for_use)
    # plt.scatter(np.where(labels == 1)[0], filtered_close_for_use[np.where(labels == 1)[0]], marker='o', c='r')
    # plt.scatter(np.where(labels == 2)[0], filtered_close_for_use[np.where(labels == 2)[0]], marker='o', c='y')
    # plt.scatter(np.where(labels == 3)[0], filtered_close_for_use[np.where(labels == 3)[0]], marker='o', c='b')
    # plt.scatter(np.where(labels == 4)[0], filtered_close_for_use[np.where(labels == 4)[0]], marker='o', c='g')

    plt.scatter(np.where(pred_int == 0)[0], close_for_use[19:][np.where(pred_int == 0)[0]], marker='o', c='g',label='0')
    plt.scatter(np.where(pred_int == 1)[0], close_for_use[19:][np.where(pred_int == 1)[0]], marker='o', c='r',label='1')
    plt.scatter(np.where(pred_int == 2)[0], close_for_use[19:][np.where(pred_int == 2)[0]], marker='o', c='r',label='2')
    plt.scatter(np.where(pred_int == 3)[0], close_for_use[19:][np.where(pred_int == 3)[0]], marker='o', c='g',label='3')
    plt.legend()
    plt.show()


    print confusion_mat