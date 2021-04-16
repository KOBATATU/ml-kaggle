import os,sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.path_setting import *
import pandas as pd
import numpy as np
import os,random


random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1234)


#columsの処理
#NA値が多い時や多くのcolumnsが多い時などを処理
def get_too_many_null_attr(data,rate = 0.8):
    #nullが90%以上のcolの名前
    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > rate]
    return many_null_cols

def get_too_many_repeated_val(data,rate = 0.8):
    big_top_value_cols = [col for col in data.columns if data[col].value_counts(dropna=False, normalize=True).values[0] > rate]
    return big_top_value_cols


def get_useless_columns(data,rate = 0.8):
    #必要ではないNA値を抽出する
    too_many_null = get_too_many_null_attr(data,rate)
    print("More than 90% null: " + str(len(too_many_null)))
    too_many_repeated = get_too_many_repeated_val(data,rate)
    print("More than 90% repeated value: " + str(len(too_many_repeated)))
    cols_to_drop = too_many_null + too_many_repeated
    return cols_to_drop



#メモリー削減
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        if i % 50 == 0:
            print(i)
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df



def load_data(csv_path,pkl_path):

    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
        return df
    else:

        df = pd.read_csv(csv_path)
        df.to_pickle(pkl_path)
    return df




