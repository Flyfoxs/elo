from code_felix.core.train import *


if __name__ == '__main__':

    for list_type in [0, 4]:
        for sn in range(0, 3):
            train, label, test = get_feature_target('1219', list_type=list_type)
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