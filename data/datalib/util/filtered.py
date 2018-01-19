# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from dateutil.parser import parse
import numpy as np

def MyFilter(data, window_width=10, beta=2.0, draw_graph=False):
    r"""Return a dataframe filtered by Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    data : pandas.dataframe(See at pandas)
        Input data. A column named 'close' are requered here.
    window_width: int
        Number of points in the window(a number less than 20 is recommended).
        If zero or less, raise Error
    beta : float
        Shape parameter, determines trade-off between main-lobe width and
        side lobe level. As beta gets large, the window narrows.
    draw_graph : bool, optional
        When True (False is defaulted), draw a graph of the original signal and
        the filtered signal,the high frequency is also drawn in another graph.

    Returns
    -------
    data_out : pandas.dataframe
        Output data.A column of filtered data is included, and the index is
        datatime form.
    """

    #read data and change the format
    if 'time' in data.columns:
        date_list = []
        for i in data.index:
            date_parse = parse(str(data.ix[i].time))
            date_list.append(date_parse)
        data['date'] = date_list
        data_use = data
        data_use.index = data_use['date'].tolist()
        data_use = data_use.drop(['date','time'], axis=1)
        data_use.index.name = 'time'
    else:
        data_use = data
    #design filter, use the kaiser window here
    window = signal.kaiser(window_width, beta=beta)
    data_use['close_filtered'] = signal.convolve(data_use['close'], window, mode='same') / sum(window)
    data_use['high_frequency'] = data_use['close'] - data_use['close_filtered']

    #delete the distortion datas after filtered
    if window_width % 2 == 0:
        data_changed = data_use[window_width/2: -(window_width/2 - 1)]
    else:
        data_changed = data_use[(window_width-1)/2: -(window_width-1)/2]

    #draw graph
    if (draw_graph == True) :
        fig = plt.figure()
        ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
        data_changed.loc[:,'close'].plot(style='r', label='original')
        data_changed.loc[:,'close_filtered'].plot(style='k', label='filtered')
        plt.title('Kaiser window_width = %d  , const = %d' % (window_width, beta))
        plt.legend(loc='best')

        ax2 = plt.subplot2grid((3,1), (2,0))
        data_changed.loc[:,'high_frequency'].plot(label='high_frequency')
        ax2.set_ylim([-150, 150])
        plt.title('High Frequency')
        plt.legend(loc='best')
        plt.show()
    # print data_use
    # print data_changed
    data_out = data_changed['close_filtered']
    return np.array(data_out.tolist())

def get_filtered_data(index_close_path,filtered_data_path):
    data_original = pd.read_csv(index_close_path, names=['one','two','three','four','five'])
    print data_original.shape
    data_use = pd.DataFrame(data_original['four'].tolist(),columns=['close'])
    print data_use.shape
    # print 'raw_data shape %i' %data_use.shape
    # 日期截选
    # data_test = data_original.loc[(data_original.time > 201308130930) & (data_original.time < 201308131400)]
    data_filtered = MyFilter(data_use, window_width=10, draw_graph=False)
    print 'filtered data shape length %i' %len(data_filtered)
    data_filtered.to_csv(filtered_data_path)
    return data_filtered

if __name__ == '__main__':
    filtered_data_path = '/home/zqfrgzn04/datasource/tmp/threeYear_BCmin_filtered_10.csv'
    index_close_path = "/home/zqfrgzn04/datasource/csvRawKData/industry_index/BioCHem.csv"
    get_filtered_data(index_close_path,filtered_data_path,draw_graph=False)