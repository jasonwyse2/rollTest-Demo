import numpy as np
import util.filtered as close_filter
import util.get_3d_data_set as get3d

import csv,os
import MySQLdb
import pandas as pd

class Raw3DimData(object):
    ''' this Class is a Generator of input data for training our model,the data shape is 28*5*time
        user just need input the code_in'''
    def __init__(self,start_Date,end_Date,index_name,code_id):
        workdirname = index_name+'-'+'ThreeDimDataForUse'+'-'+start_Date[:-4]+'-'+end_Date[:-4]
        self.current_path = os.getcwd()
        wkdir = os.path.join(self.current_path, workdirname)
        if os.path.exists(wkdir):
            os.chdir(wkdir)
        else:
            os.mkdir(wkdir)
            os.chdir(wkdir)

        self.index_name = index_name
        self.start_Date = start_Date
        self.end_Date = end_Date
        self.code_id = code_id

        self.raw_data_path = os.getcwd() +'/'+self.index_name+'-'+start_Date+'-'+end_Date+'.csv'
        self.filtered_close_path = os.getcwd() + '/' +self.index_name +'_close_filtered.csv'
        self.ratecross3ddata_path = os.getcwd() + '/' + 'ratecross3dFactors_' + self.index_name + start_Date + '-' + end_Date + '.npy'

        self.rawCloseForUse_path = os.getcwd() + '/'+self.index_name + 'RawCloseForUse.npy'
        self.rawFilteredForUse_path = os.getcwd() + '/' + self.index_name + 'FilteredDataForUse.npy'
        self.LabelsForUse_path = os.getcwd() + '/' + self.index_name + '_4Labels_unequal.npy'
        self.dataForUse_path = os.getcwd() + '/' + self.index_name + 'DataForUse.npy'
        self.timeSeriasForUse_path = os.getcwd() + '/' + self.index_name + 'TimeSeriasForUse.npy'

        self.raw_kdata = self.query_kData()

    @staticmethod
    def login_MySQL():
        db_host = '192.168.1.11'
        db_port = 3306
        db_user = 'zqfordinary'
        db_pass = 'Ab123456'
        db_name = 'stock'
        db = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name, charset='utf8')
        return db

    def query_kData(self):

        query = ("SELECT time,open,high,low,close,volume,pct_chg FROM index_min WHERE code_id=" + str(self.code_id) + " AND time>=" + self.start_Date +" AND time<=" + self.end_Date)

        db = Raw3DimData.login_MySQL()
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()

        # csvfile_name =os.getcwd() + '/hs300-'+self.start_Date+'-'+self.end_Date
        with open(self.raw_data_path,'w') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume', 'pct_chg'])
            for row in result:
                writer.writerow(row)
        # print 'exported %s !' % self.raw_data_path

        kdata = pd.read_csv(self.raw_data_path).as_matrix()

        return kdata

    def get_filtered_data(self):
        index_close = self.raw_kdata[:,4]
        close = pd.DataFrame(index_close, columns=['close'])
         # = MyFilter(data_use, window_width=10, draw_graph=False)
        filtered_data = close_filter.MyFilter(close, window_width=10, draw_graph=False)
        df = pd.DataFrame(filtered_data)
        df.to_csv(self.filtered_close_path)
        return filtered_data

    def extend_mat(self, k_data, n_kdata, N):
        ''' extend the scaled data which scaled by N minutes, extending the data everyday'''

        N_KData_1 = np.zeros((k_data.shape[0] - 241, n_kdata.shape[1]))
        N_KData_0 = np.repeat(n_kdata,N, axis=0)
        for day_num in range(k_data.shape[0]/241-1):
            N_KData_1[day_num*241:day_num*241+240,:] = N_KData_0[day_num*240:day_num*240+240]
            N_KData_1[day_num*241+240,:] = N_KData_0[day_num*240+239,:]
        return N_KData_1

    def get_n_kdata(self,close, n):
        num_day = close.shape[0] / 241
        num_n_everyday = 241 / n
        num_n_time = num_day * num_n_everyday
        n_kdata = np.zeros((num_n_time, 3))
        another3Factors = np.zeros((close.shape[0], 3))
        for day_number in range(close.shape[0] / 241):
            for j in range(int(241 / n)):
                if j == int(241 / n) - 1:
                    open_n = close[j * n + day_number * 241]
                    close_n = close[j * n + day_number * 241 + n]
                    high_n = np.max(close[(j * n + day_number * 241):(j * n + day_number * 241) + n + 1])
                    low_n = np.min(close[(j * n + day_number * 241):(j * n + day_number * 241) + n + 1])
                    # get n_kdata
                    n_kdata[j + day_number * 240 / n, 0] = (close_n - open_n) / open_n
                    n_kdata[j + day_number * 240 / n, 1] = (close_n - high_n) / high_n
                    n_kdata[j + day_number * 240 / n, 2] = (close_n - low_n) / low_n
                    # get extend factors
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n + 1, 0] = (close_n - open_n) / open_n
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n + 1, 1] = ( close_n - high_n) / high_n
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n + 1, 2] = (close_n - low_n) / low_n
                else:
                    open_n = close[(j * n + day_number * 241)]
                    close_n = close[(j * n + day_number * 241) + n - 1]
                    high_n = np.max(close[(j * n + day_number * 241):(j * n + day_number * 241) + n])
                    low_n = np.min(close[(j * n + day_number * 241):(j * n + day_number * 241) + n])
                    # get n_kdata
                    n_kdata[j + day_number * 240 / n, 0] = (close_n - open_n) / open_n
                    n_kdata[j + day_number * 240 / n, 1] = (close_n - high_n) / high_n
                    n_kdata[j + day_number * 240 / n, 2] = (close_n - low_n) / low_n
                    # get extend factors
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n, 0] = (close_n - open_n) / open_n
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n, 1] = ( close_n - high_n) / high_n
                    another3Factors[(j * n + day_number * 241):(j * n + day_number * 241) + n, 2] = ( close_n - low_n) / low_n

        return n_kdata, another3Factors

    def get_3d_raw_data(self):

        min_data = self.query_kData()[:,1:6]

        k_data = np.zeros(min_data.shape)
        volume_mean = np.mean(min_data[:,4])
        stdr_sigma = np.sqrt(np.var(min_data[:,4]))
        # print len(np.where(min_data == 0)[0])
        # print 'k_data.shape', min_data.shape
        periods = [5, 15, 30, 60]
        for i in range(len(periods) + 1):
            if i == 0:
                factors = get3d.get_all_factors(min_data[:, 0], min_data[:, 1], min_data[:, 2], min_data[:, 3], min_data[:, 4])
                factors[:,4] = (factors[:,4]-volume_mean)/stdr_sigma
                factors_stay= factors[:, [4, 8, 9, 18]]

                k_data[1:,:4] = np.diff(min_data[:, :4],axis=0)/min_data[:-1, :4]
                k_data[1:,4] = factors_stay[1:,0]
                # k_data[1,:] = 0
                factors = get3d.get_all_factors(k_data[1:, 0], k_data[1:, 1], k_data[1:, 2], k_data[1:, 3], k_data[1:, 4])
                factors[:,[4,8,9,18]] = factors_stay[1:,:]

                factors_mat = np.zeros((factors.shape[1], len(periods) + 1, factors.shape[0]-240))
                # print factors_mat
                factors_mat[:, i, :] = factors[240:,:].T
            else:
                n_kdata = get3d.get_N_KData(periods[i-1], min_data)
                factors = get3d.get_all_factors(n_kdata[:, 0], n_kdata[:, 1], n_kdata[:, 2], n_kdata[:, 3], n_kdata[:, 4])
                factors[:, 4] = (factors[:, 4] - volume_mean) / stdr_sigma
                factors_stay= factors[:, [4, 8, 9, 18]]

                n_kdata[1:,:4] =np.diff(n_kdata[:,:4],axis=0)/n_kdata[:-1, :4]
                # n_kdata[1, :] = 0
                n_kdata[1:,4] = factors_stay[1:,0]
                factors = get3d.get_all_factors(n_kdata[1:, 0], n_kdata[1:, 1], n_kdata[1:, 2], n_kdata[1:, 3], n_kdata[1:, 4])
                factors[:,[4,8,9,18]] = factors_stay[1:,:]
                factors = factors[241/periods[i-1]-1:,:]
                # print 'n_kdata.shape', n_kdata.shape
                # print 'number of zero in factors', len(np.where(factors[:,0]==0)[0])
                extend_factors = self.extend_mat(min_data, factors, periods[i-1])
                # extend_factors = extend_factors[1:,:]
                # print 'extend_factors.shape', extend_factors.T.shape
                factors_mat[:, i, :] = extend_factors.T


        # print 'number of zero in factor matrix',len(np.where(factors_mat == 0)[0])
        # print 'factors matrix shape', factors_mat.shape
        cut_num = 119*np.max(periods)
        factors_mat = factors_mat[:,:,cut_num:]
        periods = [1, 5, 15, 30, 60]

        # another three factors
        data1 = []
        close = self.raw_kdata[:,4]
        for n in periods:
            if n ==1:
                index = self.query_kData()[:,1:6]
                time_len = index.shape[0]
                one_min_3factors = np.zeros((time_len, 3))
                for i in range(time_len):
                    one_min_3factors[i, 0] = (index[i, 3] - index[i, 0])/index[i,0]
                    one_min_3factors[i, 1] = (index[i, 3] - index[i, 1])/index[i,1]
                    one_min_3factors[i, 2] = (index[i, 3] - index[i, 2])/index[i,2]
                data1.append(one_min_3factors.transpose(1,0))

            else:
                n_kdata, another3 = self.get_n_kdata(close, n)
                # print n_kdata[:, 0].shape
                # print another3.shape
                data1.append(another3.transpose(1,0))
        data1 = np.array(data1)

        # print 'data1.shape', data1.shape
        data1 = data1[:,:,241+119*60:].transpose(1,0,2)
        # print 'data1.shape', data1.shape

        data1 = np.row_stack((factors_mat[:,:,:],data1))


        revised_data_shape = (data1.shape[0], data1.shape[1], data1.shape[2]-60)
        # print 'revised_data_shape', revised_data_shape
        revised_data = np.zeros(revised_data_shape)
        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                if j ==0:
                    revised_data[i,j,:] = data1[i,j,60:]
                else:
                    revised_data[i,j,:] = data1[i,j,60-periods[j]+1:-periods[j]+1]
        # print 'revised_data.shape',revised_data.shape
        assert len(np.where(np.isnan(revised_data))[0])==0,'there exit nan in factors!'

        np.save(self.ratecross3ddata_path,revised_data)

        return revised_data

    def get_dataSets(self):

        filtered_close = self.get_filtered_data().tolist()
        # print type(filtered_close)
        t = range(len(filtered_close))
        # t, filtered_close = get3d.genData(filtered_data_path)

        knee_t, knee_close = get3d.knee_point(t, filtered_close)

        break_t, break_p = get3d.break_point(0.001, knee_t, knee_close, t, filtered_close)

        break_t = np.array(break_t)
        break_t = break_t[np.where(break_t>(241+119*60+60-5))]

        data_idx = break_t - 119 * 60 - 241 - 60 + 5 + 1
        # print 'data_idx[0]', data_idx[0]
        break_idx = data_idx -data_idx[0]
        label_idx = zip(break_idx[:-1], break_idx[1:])


        raw_3d_data = self.get_3d_raw_data()
        raw_data1 = raw_3d_data[:,:,:-4]

        data_for_use = raw_data1[:,:,data_idx[0]:data_idx[-1]]


        filtered_close_for_use = filtered_close[break_t[0]:break_t[-1]]

        close_for_use = self.raw_kdata[5:-4,4][break_t[0]:break_t[-1]]

        time_serias = self.raw_kdata[5:-4,0][break_t[0]:break_t[-1]]

        labels = np.zeros(break_idx[-1])

        if filtered_close_for_use[1] - filtered_close_for_use[0] > 0:

            for i, idx in enumerate(label_idx):

                if i % 2 == 0:
                    labels[idx[0]:idx[0]+3]= 1
                    labels[idx[0] + 3:idx[1] - 3] = 2
                    labels[idx[1] - 3:idx[1]] = 3
                else:
                    labels[idx[0]:idx[0] + 3] = 3
                    labels[idx[0]+3:idx[1] - 3] = 4
                    labels[idx[1] -3 :idx[1]] = 1

        else:

            for i, idx in enumerate(label_idx):

                if i % 2 == 0:
                    labels[idx[0]:idx[0] + 3] = 3
                    labels[idx[0] + 3:idx[1] - 3] = 4
                    labels[idx[1] - 3:idx[1]] = 1
                else:
                    labels[idx[0]:idx[0] + 3] = 1
                    labels[idx[0] + 3:idx[1] - 3] = 2
                    labels[idx[1] - 3:idx[1]] = 3

        np.save(self.LabelsForUse_path, labels)
        np.save(self.dataForUse_path, data_for_use)
        np.save(self.rawFilteredForUse_path, filtered_close_for_use)
        np.save(self.rawCloseForUse_path,close_for_use)
        np.save(self.timeSeriasForUse_path,time_serias)

        os.chdir("../")
        return data_for_use, labels,filtered_close_for_use,time_serias

def get_trainValid_datasets(code_wind,index_type,start_time,end_time):
    """ this is the main function to get the 3 Dimensoin data """
    #HS300_CODE_ID = 3037
    #ZZ500_CODE_ID = 3042

    query = ("SELECT code_id,code_wind, name from stock WHERE type=" +index_type+" AND code_wind ="+code_wind)

    db = Raw3DimData.login_MySQL()
    cursor = db.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    print result
    v = result[0]
    name = v[2]


    raw3ddata = Raw3DimData(start_time, end_time,str(v[1][:-3]),code_id=int(v[0]))
    with open('readme.txt','w') as f:
        f.write('code_id:'+str(v[1])+'---'+'wind_code:'+str(v[1])+'---'+'index_name:'+name.encode('utf8'))
        f.write('/n')

    # print raw3ddata.query_kData().shape
    data_for_use, labels, filtered_close_for_use, time_serias = raw3ddata.get_dataSets()

    return data_for_use, labels, filtered_close_for_use, time_serias


if __name__ == '__main__':
    import time

    t0 = time.time()

    index_type = '200'
    code_wind = '000001'
    start_time = '201308130930'
    end_time = '201704301500'
    data_for_use, labels, filtered_close_for_use, time_serias = get_trainValid_datasets(code_wind,index_type,start_time,end_time)
    print 'take time: ', time.time()-t0