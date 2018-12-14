
import os

import pandas as pd

from code_felix.utils_.util_log import *


class Cache_File:
    def __init__(self):
        self.df_key = 'df'
        self.cache_path='./cache/'
        self.enable=True
        self.date_list = ['start','close','start_base','weekbegin', 'tol_day_cnt_min',	'tol_day_cnt_max']
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)

    def get_path(self, key, type):
        return f'{self.cache_path}{key}.{type}'

    def readFile(self, key, file_type):
        if self.enable:
            path = self.get_path(key, file_type)
            if os.path.exists(path):

                if file_type == 'h5':
                    with pd.HDFStore(path) as store:
                        key_list = store.keys()
                    logger.debug(f"try to read cache from file:{path}, type:{file_type}, key:{key_list}")
                    if len(key_list) == 0:
                        return None
                    elif len(key_list) == 1 :
                        return pd.read_hdf(path, key_list[0])
                    else:
                        return tuple([ pd.read_hdf(path, key) for key in key_list])

            else:
                logger.debug(f"Can not find cache from file:{path}")
                return None
        else:
            logger.debug( "disable cache")


    def writeFile(self, key, val, file_type):
        if not self.enable :
            logger.debug('Cache is disable')
            return None

        if val is None or len(val)==0:
            logger.debug('Return value is None or empty')
            return val
        elif isinstance(val, tuple):
            val_tuple = val
        else:
            val_tuple = (val,)

        if all([ isinstance(item, (pd.DataFrame, pd.Series)) for item in val_tuple]) :
            path = self.get_path(key, file_type)
            if file_type == 'h5':
                for index, df in enumerate(val_tuple):
                    key = f'{self.df_key}_{index}'
                    logger.debug(f"====Write {len(df)} records to File#{path}, with:{key}")
                    df.to_hdf(path, key)
            return val
        else:
            logger.warning(f'The return is not DataFrame or it is None:{[ isinstance(item, pd.DataFrame) for item in val_tuple]}')
            return val

cache =  Cache_File()

import functools
def file_cache(overwrite=False, type='h5', prefix=None):
    """
    :param time: How long the case can keep, default is 1 week
    :param overwrite: If force overwrite the cache
    :return:
    """
    def decorator(f):
        @timed()
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            mini_args = get_mini_args(args)
            logger.debug(f'fn:{f.__name__}, para:{str(mini_args)}, kw:{str(kwargs)}')
            key = '_'.join([f.__name__, str(mini_args), str(kwargs)])
            if prefix:
                key  = '_'.join([prefix, key])
            if overwrite==False:
                val = cache.readFile(key, type)
            if overwrite==True or val is None :
                val = f(*args, **kwargs) # call the wrapped function, save in cache
                cache.writeFile(key, val, type)
            return val # read value from cache
        return wrapper
    return decorator

def get_mini_args(args):
    args_mini = [item.split('/')[-1] if isinstance(item, str) else item
                    for item in args
                        if (type(item) in (tuple, list, dict) and len(item) <= 5)
                            or type(item) not in (tuple, list, dict, pd.DataFrame)
                 ]



    df_list  =  [item for item in args if isinstance( item, pd.DataFrame) ]

    i=0
    for df in df_list:
        args_mini.append(f'df{i}_{len(df)}')
        i += 1

    return args_mini

if __name__ == '__main__':

    @timed()
    @file_cache()
    def test_cache(name):
        import time
        import numpy  as np
        time.sleep(3)
        return pd.DataFrame(data= np.arange(0,10).reshape(2,5))


    @timed()
    @file_cache()
    def test_cache_2(name):
        import time
        import numpy  as np
        time.sleep(3)
        df = pd.DataFrame(data= np.arange(0,10).reshape(2,5))
        return (df, df)


    print(test_cache('Felix'))
    print(test_cache_2('Felix'))
    #print(test_cache('Felix'))




