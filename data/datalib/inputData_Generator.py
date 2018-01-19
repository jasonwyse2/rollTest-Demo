import numpy as np
import util.get_3d_data_set as get3d
import MySQLdb
import time

class Raw3DimData(object):
    def __init__(self,index_name,code_id):

        self.index_name = index_name
        self.code_id = code_id
        self.raw_kdata = self.query_kData()

    @staticmethod
    def login_MySQL():
        """ log on databoase
        """
        db_host = '192.168.1.11'
        db_port = 3306
        db_user = 'zqfordinary'
        db_pass = 'Ab123456'
        db_name = 'stock'
        db = MySQLdb.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name, charset='utf8')
        return db

    def query_kData(self):
        """ query  minute K data for the caculation of
        """
        select_data_number = ((119*60+241+60+1)/241+1)*241
        select_data_number = str(select_data_number)
        query = ("SELECT time,open,high,low,close,volume,pct_chg FROM index_min WHERE code_id=" + str(self.code_id) +" ORDER BY id DESC LIMIT "+select_data_number)
        db = Raw3DimData.login_MySQL()
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        kdata = np.array(result)[::-1]
        return kdata

    def extend_mat(self, k_data, n_kdata, N):
        ''' extend the scaled data which scaled by N minutes, extending the data everyday'''

        N_KData_1 = np.zeros((k_data.shape[0] - 241, n_kdata.shape[1]))
        N_KData_0 = np.repeat(n_kdata,N, axis=0)
        for day_num in range(k_data.shape[0]/241-1):
            N_KData_1[day_num*241:day_num*241+240,:] = N_KData_0[day_num*240:day_num*240+240]
            N_KData_1[day_num*241+240,:] = N_KData_0[day_num*240+239,:]
        return N_KData_1

    def get_n_kdata(self,close, n):
        """get the n minute period K data"""

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
        """" Here we use the [open,high,low,close,volume] from the specific time range,we aquired 20 indicators caculated by Talib
         and 3 customed indicators, we also included the K data as indicator, finally, there are 28 indicators.
         We took [1,5,15,30,60] minute as K data unit respectively, and caculated each period indicators and extended them to the same length
         with time alignment.
        Since there exits nan at the head of the time serias, we cut them with the max index where nan appears.
        here the cut_number = (120-1)*60 + 241 + 60
        Parametres:

        """
        min_data = self.raw_kdata[:,1:6]

        k_data = np.zeros(min_data.shape)
        volume_mean = np.mean(min_data[:,4])
        stdr_sigma = np.sqrt(np.var(min_data[:,4]))
        print len(np.where(min_data == 0)[0])
        print 'k_data.shape', min_data.shape
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
                data1.append(another3.transpose(1,0))
        data1 = np.array(data1)
        data1 = data1[:,:,241+119*60:].transpose(1,0,2)
        data1 = np.row_stack((factors_mat[:,:,:],data1))
        revised_data_shape = (data1.shape[0], data1.shape[1], data1.shape[2]-60)
        revised_data = np.zeros(revised_data_shape)
        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                if j==0:
                    revised_data[i,j,:] = data1[i,j,60:]
                else:
                    revised_data[i, j,:] = data1[i, j, 60 - periods[j] + 1:-periods[j] + 1]

        return revised_data

    def get_dataSets(self):

        raw_3d_data = self.get_3d_raw_data()
        data_for_use = raw_3d_data[:,:,-1,np.newaxis]
        close_for_use = self.raw_kdata[241+119*60+60:,4][-1]
        time_serias = self.raw_kdata[241+119*60+60:,0][-1]

        return data_for_use,close_for_use,time_serias

def get_update_input(code_wind):

    #HS300_CODE_ID = 3037
    #ZZ500_CODE_ID = 3042
    t0 = time.time()
    query = ("SELECT code_id,code_wind, name from stock WHERE type=200 AND code_wind ="+code_wind)

    db = Raw3DimData.login_MySQL()
    cursor = db.cursor()
    cursor.execute(query)
    result = cursor.fetchall()

    v = result[0]

    raw3ddata = Raw3DimData(str(v[1][:-3]),code_id=int(v[0]))

    data_for_use,close_for_use,time_serias = raw3ddata.get_dataSets()
    print 'data_for_use,close_for_use,time_serias',data_for_use[3,0,0],close_for_use,time_serias
    print 'take time: ', time.time()-t0
    return data_for_use,close_for_use,time_serias

if __name__ =='__main__':

    code_wind = '000905'
    data_for_use, close_for_use, time_serias = get_update_input(code_wind)