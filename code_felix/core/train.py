from code_felix.feature.read_file import *
from code_felix.core.sk_model import *


# params = {'num_leaves': 54,
#          'min_data_in_leaf': 79,
#          'objective': 'regression',
#          'max_depth': 7,
#          'learning_rate': 0.018545526395058548,
#          "boosting": "gbdt",
#          "feature_fraction": 0.8354507676881442,
#          "bagging_freq": 3,
#          "bagging_fraction": 0.8126672064208567,
#          "bagging_seed": 11,
#          "metric": 'rmse',
#          "lambda_l1": 0.1,
#          "verbosity": -1,
#          'min_child_weight': 5.343384366323818,
#          'reg_alpha': 1.1302650970728192,
#          'reg_lambda': 0.3603427518866501,
#          'subsample': 0.8767547959893627,}


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
         "reg_alpha": 0.2634,
         #"reg_lambda":
         "random_state": 133,
         "verbosity": -1,
         "verbose":-1, #No further splits with positive gain
         }


if __name__ == '__main__':
    for version in [ ('1215')]:

        train, label, test = get_feature_target(version)
        logger.debug(f'get_feature_target result:{train.shape}, {label.shape}, {test.shape}')

        model_type = 'lgb'
        oof, prediction, score = train_model(train, test, label, params=params,  model_type=model_type, plot_feature_importance=False)

        des = '{0:.6f}_{1}_{2}({3})'.format( score, model_type, get_params_summary(params), train.shape[1] )

        sub_df = pd.DataFrame({"card_id":test.index})
        sub_df["target"] = prediction
        file = "./output/submit_{0}_{1}.csv".format(des, version)
        sub_df.to_csv(file, index=False)
        logger.debug(f'Sub file save to :{file}')





