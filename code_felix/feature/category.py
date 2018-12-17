from code_felix.feature.read_file import _get_transaction, cal_ratio
from code_felix.feature.read_file import *
from code_felix.utils_.util_pandas import flat_columns
from code_felix.utils_.util_log import logger

import pandas as pd
def get_cat_feature(file):
    if isinstance(file, pd.DataFrame):
        df = file
    else:
        df = _get_transaction(file)
    tmp = df.groupby(['card_id', 'merchant_category_id']).agg({'purchase_amount': ['sum', 'count'], })
    tmp = flat_columns(tmp)
    tmp = tmp.reset_index()
    tmp = tmp.pivot(index='card_id', columns='merchant_category_id', values='purchase_amount_count')
    tmp.fillna(0, inplace=True)
    return tmp


@lru_cache()
def get_cat_ratio(base=trans_his_file, new=trans_new_file):
    base = get_cat_feature(base)
    new = get_cat_feature(new)

    logger.debug(f'New have more column:{[ col for col in new.columns if col not in base.columns]}')

    common_cols = [col for col in base.columns if col in new.columns]
    logger.debug(f'Common column list:{common_cols}')
    new = new[common_cols]
    base = base[common_cols]
    return cal_ratio(base,new, 'cat_vs')
