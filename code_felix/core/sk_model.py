from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from code_felix.utils_.util_cache_file import *
import numpy as np
import time
import pandas as pd

import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from code_felix.utils_.util_pandas import save_result_for_ensemble
from code_felix.core.dnn_model import *


def get_params_summary(params):
    key_list = ['num_leaves', 'max_depth', 'lambda_l1']

    params = [ f'{key}={params[key]}'  for key in key_list if key in params]

    return '_'.join(params)



@timed()
def train_model(X, X_test, y, params,  model_type='lgb', plot_feature_importance=False):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=777)

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        logger.debug(f'Fold:{fold_n}, started at:{time.ctime()}')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators=20000, nthread=4, n_jobs=-1)
            model.fit(X_train, y_train,
                      eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                      verbose=True, early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=50, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

        if model_type == 'rcv':
            model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=3)
            model.fit(X_train, y_train)
            logger.debug(model.alpha_)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000, eval_metric='RMSE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        if model_type.startswith('dnn'):

            from multiprocessing import Process, Queue

            @timed()
            def fn( X_train, y_train, X_valid, y_valid, queue):
                model = ELO_model(X_train.shape[1], model_type=model_type)
                model.fit(X_train, y_train, X_valid, y_valid)

                y_pred_valid = model.predict(X_valid)
                y_pred = model.predict(X_test)

                queue.put((y_pred_valid, y_pred))

            queue = Queue()

            p = Process(target=fn, args=(X_train, y_train, X_valid, y_valid, queue))
            p.start()
            p.join()
            y_pred_valid, y_pred = queue.get()


            logger.debug(f'y_pred_valid:{y_pred_valid.shape}, y_pred:{y_pred.shape}')

            #logger.debug(f'result len, ID:{id(result)}')

        oof[valid_index] = y_pred_valid.reshape(-1, )
        score = mean_squared_error(y_valid, y_pred_valid) ** 0.5
        scores.append(score)

        logger.debug('Folder:{}, score:{}'.format(fold_n, score) )

        logger.debug(f'prediction:{prediction.shape}, y_pred:{y_pred.shape}')
        prediction += y_pred

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    logger.debug('CV mean score: {0:.4f}, std: {1:.4f}. score list:{2}'.format(np.mean(scores), np.std(scores), scores))

    score_avg = round(np.mean(scores), 7)
    oof = pd.DataFrame(oof, columns=['train'], index=X.index)
    y = pd.DataFrame(y.values, columns=['y'], index=X.index)
    prediction = pd.DataFrame(prediction, columns=['prediction'], index=X_test.index)

    if score_avg <= 3.63:
        save_result_for_ensemble(model_type, file_name=f'{model_type}_{score_avg}', train=oof, label=y, prediction=prediction)

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            return oof, prediction, feature_importance

    return oof, prediction, score_avg



if __name__ == '__main__':
    from code_felix.core.train import param
    logger.debug(get_params_summary(param))