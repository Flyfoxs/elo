from keras import Sequential, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam

from code_felix.core.params import get_model_paras
from code_felix.core.sk_model import train_model
from code_felix.utils_.util_log import *
import numpy as np

def get_all_file(path):
    import os
    logger.debug(f'Try to read file from"{path}')
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.h5' in file]
    file_list = sorted(file_list)
    return file_list



def get_top_n(folder, top=1):
    result = get_all_file(folder)
    result = result[:top]
    logger.debug(f'Pick {len(result)} file from {folder}: file:{result}')
    return [f'{folder}/{file}' for file in result]

# file_list  = get_top_n('./output/1level/xgb', 3)


def get_stacking_feature(file_list):
    train = [pd.read_hdf(file, 'train') for file in file_list]

    label =  pd.read_hdf(file_list[0],'label')

    test =  [pd.read_hdf(file, 'prediction') for file in file_list]

    return pd.concat(train, axis=1), label.loc[:,'y'], pd.concat(test, axis=1)


if __name__ == '__main__':

    for sn in range(0, 10):

        file_list = get_top_n('./output/1level/xgb', top=3)
        file_list.extend(get_top_n('./output/1level/lgb', top=3))

        train, label, test =  get_stacking_feature(file_list)
        logger.debug(f'get_feature_target result:{train.shape}, {label.shape}, {test.shape}')

        model_type = f'dnn{sn}'
        model_paras = get_model_paras(model_type)
        oof, prediction, score = train_model(train, test, label, model_paras, model_type=model_type, )

        des = '{0:.6f}_{1}_{2}({3})'.format(score, model_type, 'stacking', train.shape[1])

        sub_df = pd.DataFrame(index=test.index)
        sub_df["target"] = prediction.iloc[:, 0].values
        file = "./output/2level/submit_{0}_{1}_{2}.csv".format(des, 'stacking', model_type)
        sub_df.to_csv(file, index=True)
        logger.debug(
            f'Sub file save to :{file}, With stacking model paras:default')





