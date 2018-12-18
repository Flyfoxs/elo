from code_felix.feature.read_file import *
from code_felix.core.sk_model import *
from code_felix.utils_.other import *
from code_felix.core.params import *


def gen_sub(model_args, model_type, version=None, list_type=None, ):


    logger.debug(f'Model({model_type}) params:{params}')
    if version is None:
        version = '1215'

    train, label, test = get_feature_target(version, list_type=list_type)
    logger.debug(f'get_feature_target result:{train.shape}, {label.shape}, {test.shape}')

    model_paras = get_model_paras(model_type, args)
    oof, prediction, score = train_model(train, test, label, params=model_paras, model_type=model_type,)

    des = '{0:.6f}_{1}_{2}({3})'.format(score, model_type, get_params_summary(model_paras), train.shape[1])

    sub_df = pd.DataFrame({"card_id": test.index})
    sub_df["target"] = prediction
    file = "./output/submit_{0}_{1}.csv".format(des, version)
    sub_df.to_csv(file, index=False)
    logger.debug(
        f'Sub file save to :{file}, With model paras:{get_pretty_info(model_args)}, version:{version}, list_type:{list_type}')



if __name__ == '__main__':
    #for version in [ ('1215')]:
   for args in [
         # {'feature_fraction': 0.4, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 90},
         # {'feature_fraction': 0.5, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 90},
         # {'feature_fraction': 0.4, 'max_depth': 9, 'reg_alpha': 0.8, 'reg_lambda': 90}, #F
         # {'feature_fraction': 0.4, 'max_depth': 8, 'reg_alpha': 1.0, 'reg_lambda': 90},
         # {'feature_fraction': 0.4, 'max_depth': 8, 'reg_alpha': 2.0, 'reg_lambda': 90},
         # {'feature_fraction': 0.4, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 150}, #F
         # {'feature_fraction': 0.4, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 200},



       {'feature_fraction': 0.7, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 90},
       {'feature_fraction': 0.65, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 90},
       {'feature_fraction': 0.7, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 200},
       {'feature_fraction': 0.65, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 200},
       {'feature_fraction': 0.7, 'max_depth': 9, 'reg_alpha': 0.8, 'reg_lambda': 200},
       {'feature_fraction': 0.7, 'max_depth': 9, 'reg_alpha': 1, 'reg_lambda': 200},
       {'feature_fraction': 0.7, 'max_depth': 9, 'reg_alpha': 2, 'reg_lambda': 200},
       {'feature_fraction': 0.7, 'max_depth': 9, 'reg_alpha': 3, 'reg_lambda': 200},
   ]:
       model_type = 'xgb'
       gen_sub(args, model_type)


