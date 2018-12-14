from code_felix.feature.read_file import *
from code_felix.core.sk_model import *


params = {'num_leaves': 54,
         'min_data_in_leaf': 79,
         'objective': 'regression',
         'max_depth': 7,
         'learning_rate': 0.018545526395058548,
         "boosting": "gbdt",
         "feature_fraction": 0.8354507676881442,
         "bagging_freq": 3,
         "bagging_fraction": 0.8126672064208567,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "verbosity": -1,
         'min_child_weight': 5.343384366323818,
         'reg_alpha': 1.1302650970728192,
         'reg_lambda': 0.3603427518866501,
         'subsample': 0.8767547959893627,}


params = {'num_leaves': 111,
         'min_data_in_leaf': 149,
         'objective':'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2634,
         "random_state": 133,
         "verbosity": -1,
         "verbose":-1, #No further splits with positive gain
         }


if __name__ == '__main__':
    train, label, test = get_feature_target()
    logger.debug(train.shape, label.shape, test.shape)
    oof, prediction, des = train_model(train, test, label, params=params,  model_type='lgb', plot_feature_importance=False)

    sub_df = pd.DataFrame({"card_id":test.index})
    sub_df["target"] = prediction
    file = "./output/submit_{}.csv".format(des)
    sub_df.to_csv(file, index=False)
    logger.debug(f'Sub file save to :{file}')

