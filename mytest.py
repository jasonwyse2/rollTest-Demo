import pandas as pd
import numpy as np
if __name__ == '__main__':
    train_confu_matrix_df_all = pd.DataFrame()
    train_confu_matrix_df = pd.DataFrame(np.arange(16).reshape(-1,4),columns = ['pred 0','pred 1', 'pred 2', 'pred 3'] ,
                                             index = ['real 0', 'real 1', 'real 2', 'real 3'])
    train_confu_matrix_df_all = pd.concat([train_confu_matrix_df_all, train_confu_matrix_df])
    print train_confu_matrix_df_all
