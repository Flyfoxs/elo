from code_felix.core.train import *


if __name__ == '__main__':
    #for version in [ ('1215')]:
   for args in [
       {'feature_fraction': 0.7, 'max_depth': 8, 'reg_alpha': 0.8, 'reg_lambda': 200},
   ]:
       for list_type in [ 0, 4]:
            gen_sub(args, 'xgb', version='compare', list_type=list_type)


