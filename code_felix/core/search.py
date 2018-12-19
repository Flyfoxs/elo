from code_felix.core.train import *
from code_felix.core.params import *
from code_felix.feature.read_file import *
from code_felix.core.sk_model import *
from code_felix.utils_.other import get_pretty_info

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

@timed()
def optimize_fun(args, model_type):
    args_input = locals()
    max_deep = args['max_depth']
    feature_fraction = args['feature_fraction']
    reg_alpha = args['reg_alpha']
    reg_lambda = args['reg_lambda']
    list_type = args['list_type']

    version = '1219'
    train, label, test = get_feature_target(version, list_type=list_type)
    logger.debug(f'{train.shape}, {label.shape}, {test.shape}')

    model_paras = get_model_paras(model_type, args)
    oof, prediction, score = train_model(train, test, label, params=model_paras, model_type=model_type,
                                         plot_feature_importance=False)
    logger.debug('Search: get {0:,.7f} base on {1},{2},feature:{3}'.format(score, model_type, args_input, train.shape[1]))

    if score <= 3.653:
        des = '{0:.6f}_{1}_{2}({3})'.format(score, model_type, get_params_summary(model_paras), train.shape[1])
        sub_df = pd.DataFrame({"card_id": test.index})
        sub_df["target"] = prediction
        file = "./output/submit_{0}_{1}.csv".format(des, version)
        sub_df.to_csv(file, index=False)
        logger.debug(f'Sub file save to :{file}, With model paras:{get_pretty_info(args)}, model:{model_type}, version:{version}, list_type:{list_type}')

    return {
        'loss': score,
        'status': STATUS_OK,
        # -- store other results like this
        #'eval_time': time.time(),
        #'other_stuff': {'type': None, 'value': [0, 1, 2]},
        # -- attachments are handled differently
        'attachments': {"message": f'{args_input},{version}', }
        }



if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        max_evals = int(sys.argv[2])

    else:
        model_type = 'lgb'
        max_evals = 2

    logger.debug(f'Try to search paras base on model:{model_type}, max_evals:{max_evals}')

    from functools import partial
    optimize_fun_ex = partial(optimize_fun, model_type=model_type)


    trials = Trials()
    space = get_search_space(model_type)
    best = fmin(optimize_fun_ex, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    #logger.debug(f"Best: {best}")

    att_message = [trials.trial_attachments(trial)['message'] for trial in trials.trials]
    for score, para, misc in zip( trials.losses() ,
                                  att_message,
                                  [item.get('misc').get('vals') for item in trials.trials]
                                  ):
        logger.debug(f'score:{"%9.6f"%score}, para:{para}, misc:{misc}')