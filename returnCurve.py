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
def get_return(save_result_dict,directory):
    # save_result_dict = {'test_y_true_all': list(test_y_true_all.tolist()),
    #                     'test_y_predict_all': list(test_y_predict_all.tolist()),
    #                     'test_close_all': list(test_close_all.tolist()), 'test_date_all': list(test_date_all.tolist())}
    test_y_true_all = save_result_dict['test_y_true_all']
    test_y_predict_all = save_result_dict['test_y_predict_all']
    test_close_all = save_result_dict['test_close_all']
    test_date_all = save_result_dict['test_date_all']
    asset = [1.0]
    short_position = 0.0
    long_position = 0.0
    amount = []
    index_list = []
    label_type = {'bottom': 0, 'up': 1, 'top': 2, 'down': 3}
    # for i in range(len(test_close_all)):
    #     if i==0:
    #         while test_y_predict_all[i]==label_type['bottom'] or test_y_predict_all[i]==label_type['top']:
    #             i = i+1
    #         if i>=len(test_close_all):
    #             break
    #         if test_y_predict_all[i] == label_type['up']:
    #             long_position = asset/test_close_all[i]
    #             previous_state = label_type['up']
    #             strike_price = test_close_all[i]
    #             amount.append(asset[-1])
    #             index_list.append(i)
    #         else:#test_y_predict_all[i] == label_type['down']:
    #             short_position = asset / test_close_all[i]
    #             previous_state = label_type['down']
    #             strike_price = test_close_all[i]
    #             amount.append(asset[-1])
    #             index_list.append(i)
    #     else:
    #         if test_y_predict_all[i]==label_type['bottom']:
    #             if previous_state != label_type['bottom']:
    #                 short_position, asset, amount = close_short_position(short_position, strike_price, test_close_all[i], asset, amount)
    #                 previous_state = label_type['bottom']
    #                 index_list.append(i)
    #         elif test_y_predict_all[i]==label_type['up']:
    #             if previous_state != label_type['up']:
    #                 short_position, asset, amount = close_short_position(short_position, strike_price, test_close_all[i], asset, amount)
    #                 long_position, asset, amount = open_long_position(long_position, test_close_all[i], asset, amount)
    #                 previous_state = label_type['up']
    #                 index_list.append(i)
    #         elif test_y_predict_all[i]==label_type['top']:
    #             if previous_state != label_type['top']:
    #                 long_position, asset, amount = close_long_position(long_position, strike_price, test_close_all[i], asset, amount)
    #                 previous_state = label_type['top']
    #                 index_list.append(i)
    #         elif test_y_predict_all[i]==label_type['down']:
    #             if previous_state != label_type['down']:
    #                 long_position, asset, amount = close_long_position(long_position, strike_price, test_close_all[i],
    #                                                                asset, amount)
    #                 short_position, asset, amount = open_short_position(short_position, strike_price, asset, amount)
    #                 previous_state = label_type['down']
    #                 index_list.append(i)
    #         else:
    #             raise Exception('unknown label_type %s'%(str(test_y_predict_all[i])))
    #
    # if previous_state == label_type['top'] or previous_state == label_type['down']:
    #     short_position, asset, amount = close_short_position(short_position, strike_price, test_close_all[-1], asset, amount)
    # else:#previous_state == label_type['bottom'] or previous_state == label_type['up']:
    #     long_position, asset, amount = close_long_position(long_position, strike_price, test_close_all[-1], asset,
    #                                                      amount)

    fig_path = directory+'asset-amount.pdf'
    y_list = [asset, amount]
    title_list=['asset','amount']
    with PdfPages(fig_path) as pdf:
        plt.figure(figsize=(40, 25))#
        figure_position = [211,212]
        for i in range(len(y_list)):
            ax = plt.subplot(figure_position[i])
            plt.sca(ax)
            y = y_list[i]
            plt.title(title_list[i],fontsize=20)
            plt.plot(np.array(index_list), np.array(y))
            #plt.scatter(range(len(y)), np.array(y))
            plt.grid()
        pdf.savefig()
        plt.close()

    return asset, amount

def close_short_position(short_position, strike_close_price, current_price, asset, amount):
    asset.append(asset[-1] + short_position * (current_price - strike_close_price))
    amount.append(amount[-1] + abs(short_position*current_price))
    short_position = 0.0
    return short_position, asset, amount
def close_long_position(long_position, strike_price, current_price, asset, amount):
    asset.append(asset[-1] + long_position * (current_price - strike_price))
    amount.append(amount[-1] + abs(long_position*current_price))
    long_position = 0.0
    return long_position, asset, amount
def open_short_position(short_position, strike_price, asset, amount):
    amount.append(amount[-1] + asset[-1])
    asset.append(asset[-1])
    short_position = short_position + (-asset[-1] / strike_price)
    return short_position, asset, amount
def open_long_position(long_position, strike_price, asset, amount):
    amount.append(amount[-1] + asset[-1])
    asset.append(asset[-1])
    long_position = long_position + asset[-1] / strike_price
    return long_position, asset, amount
