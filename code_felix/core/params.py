
from code_felix.utils_.other import get_gpu_paras
from code_felix.utils_.util_log import *


try:
    from hyperopt import hp
except Exception as e:
    logger.exception(e)

def get_model_paras(model_type, ext_paras={}):
    if model_type == 'xgb':
        return _get_xgb_paras(ext_paras)
    elif model_type == 'lgb':
        return _get_lgb_paras(ext_paras)
    elif model_type.startswith( 'dnn'):
        return model_type
    else:
        raise Exception(f'Unknown model:{model_type}')


def _get_lgb_paras(input={}):
    params = {'num_leaves': 111,
              'min_data_in_leaf': 149,
              'objective': 'regression',
              'max_depth': 8,
              'learning_rate': 0.005,
              "boosting": "gbdt",
              "feature_fraction": 0.8,
              "bagging_freq": 1,
              "bagging_fraction": 0.7083,
              "bagging_seed": 11,
              "metric": 'rmse',
              "reg_alpha": 0.8,
              "reg_lambda": 200,
              "random_state": 133,
              "verbosity": -1,
              "verbose": -1,  # No further splits with positive gain
              }
    return dict(params, **input)



def _get_xgb_paras(input={}):

    params = {
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'max_depth': 8,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'n_estimators': 20000,
    'learning_rate': 0.01,

    'seed': 1,
    'missing': None,

    # Useless Paras
    'silent': True,
    'gamma': 0,
    'max_delta_step': 0,
    'min_child_weight': 1,
    'colsample_bylevel': 1,
    'scale_pos_weight': 0.95,
              }

    input = dict([ (k, v)  for k, v in input.items() if k in params])

    gpu_paras = get_gpu_paras('xgb')

    if gpu_paras:
        gpu_ex = {'objective':'gpu:reg:linear'}
    else:
        gpu_ex = {}

    return dict(params, **input, **gpu_paras, **gpu_ex)


def get_search_space(model_type):
    if model_type == 'xgb':
        space = {"max_depth": hp.choice("max_depth", [5, 6, 7]),
                 'reg_alpha': hp.choice("reg_alpha", [0.6, 0.7, 0.8 ]),
                 'reg_lambda': hp.choice("reg_lambda", [200, 250, 300, 350]),
                 'feature_fraction': hp.choice("feature_fraction", [0.4, 0.45,0.6, 0.7]),
                 'list_type': hp.choice("list_type", [0,9, 1,2]),
                 }

    elif model_type =='lgb':
        space = {"max_depth": hp.choice("max_depth", [7,8]),
                 'reg_alpha': hp.choice("reg_alpha", [0.8]),
                 'reg_lambda': hp.choice("reg_lambda", [190, 200, 210]),
                 'feature_fraction': hp.choice("feature_fraction", [0.75, 0.8, 0.85]),
                 'list_type': hp.choice("list_type", range(0, 10)),
                 }
    else:
        space = {}
    logger.debug(f'The search space for {model_type} is {space}')

    return space
