from code_felix.feature.read_file import *
from code_felix.core.sk_model import *

from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK

@timed()
def optimize_fun(args):
    args_input = locals()
    max_deep = args['max_depth']
    feature_fraction = args['feature_fraction']
    reg_alpha = args['reg_alpha']
    reg_lambda = args['reg_lambda']
    params = {'num_leaves': 111,
             'min_data_in_leaf': 149,
             'objective':'regression',
             'max_depth': max_deep,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": feature_fraction,
             "bagging_freq": 1,
             "bagging_fraction": 0.7083 ,
             "bagging_seed": 11,
             "metric": 'rmse',
             "reg_alpha": reg_alpha,
             "reg_lambda": reg_lambda,
             "random_state": 133,
             "verbosity": -1,
             "verbose":-1, #No further splits with positive gain
             }
    version = '201814'
    train, label, test = get_feature_target(version)
    logger.debug(f'{train.shape}, {label.shape}, {test.shape}')

    model_type = 'lgb'
    oof, prediction, score = train_model(train, test, label, params=params, model_type=model_type,
                                         plot_feature_importance=False)
    logger.debug('Search: get {0:,.7f} base on {1}'.format(score,args_input))

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
        max_evals = int(sys.argv[1])
    else:
        max_evals = 2

    space = {"max_depth":      hp.choice("max_depth", range(8,11)),
             'reg_alpha':  hp.choice("reg_alpha",  np.arange(0.1, 2, 0.1).round(2)),
             'reg_lambda': hp.choice("reg_lambda", np.arange(60, 200, 10)),
             'feature_fraction': hp.choice("feature_fraction", np.arange(0.1, 1, 0.1).round(2)),
             #"num_round": hp.choice("n_estimators", range(30, 100, 20)),  # [0,1,2,3,4,5] -> [50,]
             #"threshold": hp.choice("threshold", range(300, 500, 50))
             #"threshold": hp.randint("threshold", 400),
             }

    trials = Trials()
    best = fmin(optimize_fun, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    logger.debug(f"Best: {best}")

    att_message = [trials.trial_attachments(trial)['message'] for trial in trials.trials]
    for score, para, misc in zip( trials.losses() ,
                                  att_message,
                                  [item.get('misc').get('vals') for item in trials.trials]
                                  ):
        logger.debug(f'score:{"%9.6f"%score}, para:{para}, misc:{misc}')