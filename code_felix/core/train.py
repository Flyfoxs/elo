from code_felix.feature.read_file import *
from code_felix.core.sk_model import *
from code_felix.utils_.other import *
from code_felix.core.params import *


def gen_sub(model_args, model_type, version=None, list_type=None, ):


    logger.debug(f'Model({model_type}) params:{model_args}')
    if version is None:
        version = '1215'

    train, label, test = get_feature_target(version, list_type=list_type)
    logger.debug(f'get_feature_target result:{train.shape}, {label.shape}, {test.shape}')

    model_paras = get_model_paras(model_type, model_args)
    oof, prediction, score = train_model(train, test, label, params=model_paras, model_type=model_type,)

    des = '{0:.6f}_{1}_{2}({3})'.format(score, model_type, get_params_summary(model_paras), train.shape[1])

    sub_df = pd.DataFrame({"card_id": test.index})
    sub_df["target"] = prediction.iloc[:,0]
    file = "./output/submit_{0}_{1}.csv".format(des, version)
    sub_df.to_csv(file, index=False)
    logger.debug(
        f'Sub file save to :{file}, With model paras:{get_pretty_info(model_args)}, version:{version}, list_type:{list_type}')



if __name__ == '__main__':

       model_type = 'lgb'
       gen_sub({}, model_type)

       model_type = 'xgb'
       gen_sub({}, model_type)


