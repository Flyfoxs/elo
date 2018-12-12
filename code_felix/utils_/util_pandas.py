
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder


from code_felix.utils_.util_log import *
import matplotlib as plt
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
except Exception as e:
    logger.warning(e)


@timed(logger)
def convert_label_encode(sample, exclude=[]):
    try:
        #Label encode
        obj_col = sample.select_dtypes(include=['object']).columns
        obj_col = [ item for item in obj_col if item != 'device' and item not in exclude]
        print(f'{obj_col} will convert to label encode, and fillna with Other')


        sample = sample.apply(lambda x: x.fillna('Other')
                                if x.name in obj_col else x,
                                                      reduce=False)

        label_encode = defaultdict(LabelEncoder)
        sample = sample.apply(lambda x: label_encode[x.name].fit_transform(x.astype(str))
                        if x.name in obj_col else x,
                        reduce=False)


        return sample
    except Exception as e:
        print(f'The columns typs is {sample.dtypes.sort_values()}')
        raise e

def convert_cat_2_code(df):
    for col in df:
        if df[col].dtype.name == 'category':
            df[col] = df[col].cat.codes
    return df


import numpy as np
def check_exception(df, index=None):
    df = df.copy(deep=True)
    if index is not None and index in df:
        df.set_index(index,inplace=True)
    df = df.select_dtypes( #include=['float64', 'int'],
                           exclude=['object', 'datetime64[ns]'],)
    try:
        x, y = np.where(np.isinf(df.values) | np.isnan(df.values))
    except Exception as error:
        logger.debug(df.dtypes.sort_values())
        raise error
    if len(x)>0:
        #print(x.min(), x.max()+1, y.min(), y.max()+1)
        df = df.iloc[x.min():(x.max()+3), y.min():(y.max()+3)]
        error_part = df.iloc[:3, :4]
        logger.debug(f'check_exception:\n{error_part}')
        return error_part
    else:
        return pd.DataFrame()


def flat_columns(df, prefix=None):
    df.columns = ['_'.join(drop_empty_item(item)) for item in df.columns]
    if prefix is not None:
        df.columns = [f'{prefix}_{item}' for item in df.columns]
    return df

def drop_empty_item(item):
    return [str(val) for val in item if val is not None and len(str(val))>0]

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
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
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
