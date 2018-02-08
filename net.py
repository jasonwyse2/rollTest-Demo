import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
import pymysql
import scipy.signal as signal
from numpy import random
from pandas import DataFrame
import datetime
import time
def get_net_yield(save_result_dict, directory, underlying, interval=10):
    test_y_true_all = save_result_dict['test_y_true_all']
    test_y_predict_all = save_result_dict['test_y_predict_all']
    test_close_all = save_result_dict['test_close_all']
    test_date_all = save_result_dict['test_date_all']
    label_type = {'bottom': 0, 'up': 1, 'top': 2, 'down': 3}
    #label_type = {'sharp up': 0, 'sharp down': 1, 'gentle up': 2, 'gentle down': 3} #'sharp up', 'sharp down', 'gentle up', 'gentle down'
    position_signal = np.array(test_y_predict_all[:-1])

    for i in range(len(test_y_predict_all[:-1])):
        if test_y_predict_all[i] == label_type['bottom'] and i-1>=0 and test_y_predict_all[i-1] == label_type['up']:
            position_signal[i] = label_type['up']
        elif test_y_predict_all[i] == label_type['top'] and i-1>=0 and test_y_predict_all[i-1] == label_type['down']:
            position_signal[i] = label_type['down']
    position_signal[np.where(position_signal == label_type['bottom'])] = 0
    position_signal[np.where(position_signal == label_type['top'])] = 0
    position_signal[np.where(position_signal == label_type['up'])] = 1
    position_signal[np.where(position_signal == label_type['down'])] = -1


    test_close_return = np.diff(test_close_all) / test_close_all[:-1]
    test_close_baseline_return = test_close_all[1:] / test_close_all[0]

    test_close_return_product = position_signal * test_close_return
    test_close_return_product_pulsOne = test_close_return_product + 1
    cum_prod_list = [1]  # np.cumprod(test_close_return_product_pulsOne)
    tmp_prod = 1
    for i in range(len(test_close_return_product_pulsOne)):
        tmp_prod = tmp_prod * test_close_return_product_pulsOne[i]
        cum_prod_list.append(tmp_prod)

    fig_path = directory + 'net_profit.pdf'
    with PdfPages(fig_path) as pdf:
        plt.figure(figsize=(30, 15))
        plt.title('%s Net yield (Time:%s-%s)' % (underlying, test_date_all[0], test_date_all[-1]), fontsize=20)

        y = np.array(cum_prod_list)
        plt.plot(y, label='yours')
        plt.scatter(np.arange(0,y.shape[0]), y, marker='o')
        y = np.array([1]+test_close_baseline_return.tolist())
        plt.plot(y, label='baseline')
        plt.scatter(np.arange(0,y.shape[0]), y, marker='D')
        interval_idx_list = []
        interval_date_list = []
        for i in range(len(test_close_all)):
            if i % interval == 0:
                interval_idx_list.append(i)
                interval_date_list.append(test_date_all[i])

        plt.xticks(tuple(interval_idx_list), tuple(interval_date_list))
        plt.xticks(rotation=80, fontsize=15)
        plt.grid()
        plt.legend(fontsize=25)
        pdf.savefig()
        plt.close()

if __name__ == '__main__':
    y = range(10)
    y_array = np.where(y==y)
    plt.scatter(np.arrange(0, len(y)), np.array(y), marker='o')