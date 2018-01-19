#coding:utf-8
import pymysql
import pandas as pd

def login_MySQL(db_parameter):
    db_host = '192.168.1.11'
    db_port = 3306
    db_user = 'zqfordinary'
    db_pass = 'Ab123456'
    db_name = 'stock'
    conn = pymysql.connect(host=db_host, port=db_port, user=db_user, passwd=db_pass, db=db_name)
    return  conn

def query_kdata(*args):


    conn = login_MySQL()
    sql = args[0]
    df = pd.read_sql(sql, conn, )
    result = pd.DataFrame(df)
    # query = (args[0])
    #cursor = conn.cursor()
    #cursor.execute(query)
    # cursor.execute("DROP TABLE IF EXISTS test")#必须用cursor才行
    # result = cursor.fetchall()
    #result = result.set_index('DATE')
    #cursor.close()

    return result
    # csvfile_name =os.getcwd() + '/hs300-'+self.start_Date+'-'+self.end_Date
    # with open(self.raw_data_path,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume', 'pct_chg'])
    #     for row in result:
    #         writer.writerow(row)
    # print 'exported %s !' % self.raw_data_path


    #kdata = pd.read_csv(self.raw_data_path).as_matrix()



if __name__ =='__main__':

     kdata = query_kdata('SELECT DATE,OPEN,high,low,CLOSE FROM index_date WHERE DATE>=20130101 AND code_id = 3042')
     print 'kdata'
     print kdata