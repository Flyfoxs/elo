from code_felix.utils_.util_pandas import *
from code_felix.utils_.util_cache_file import *
from functools import lru_cache
from code_felix.utils_.reduce_mem import *
trans_new_file = './input/new_merchant_transactions.csv'
trans_his_file = './input/historical_transactions.csv'

try:
    from code_felix.feature.config_local import *
except Exception as e:
    logger.warning("Import local config failed")
    logger.exception(e)

merchants_file = './input/merchants.csv'

train_file = './input/train.csv'

test_file = './input/test.csv'



@timed()
@lru_cache()
@reduce_mem()
def _get_transaction(file):

    trans =  pd.read_csv(file, parse_dates=['purchase_date'])

    trans['month_diff'] = ((pd.to_datetime('today') - trans['purchase_date']).dt.days) // 30
    trans['month_diff'] += trans['month_lag']

    merchants = pd.read_csv(merchants_file)

    merchants.columns = [ f'mer_{col}' if col != 'merchant_id' else col for col in merchants.columns ]

    return  pd.merge(trans, merchants, how='left', on='merchant_id' )



@timed()
@reduce_mem()
def get_train_test(file):
    df =  pd.read_csv(file, parse_dates=['first_active_month'],index_col='card_id')
    #df.first_active_month.fillna(pd.to_datetime('2017-03-09'), inplace=True)
    df.loc[df.index == 'C_ID_c27b4f80f7', 'first_active_month'] = pd.to_datetime('2017-03-09')
    date_parts = ["year", "weekday", "month"]
    for part in date_parts:
        part_col = 'first_active_month' + "_" + part
        df[part_col] = getattr(df['first_active_month'].dt, part).astype(int)

    max_date = pd.to_datetime('2018-02-01')
    df['elapsed_time'] = (max_date - df['first_active_month']).dt.days

    del df['first_active_month']
    return df


@timed()
def _summary_card_trans_col(df, agg_fun):


    df['authorized_flag'] = df['authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)

    df['purchase_month']  = df['purchase_date'].dt.month

    #logger.debug(f'category_1:{df.category_1.dtype.name}')
    if df.category_1.dtype.name == 'object':
        map_dict = {'Y': 0, 'N': 1}
        df['category_1'] = df['category_1'].apply(lambda x: map_dict[x]).astype('int8')

    # map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3, np.nan: 3}
    # df['category_3'] = df['category_3'].apply(lambda x: map_dict[x])

    df = pd.get_dummies(df, columns=['category_2', 'category_3'])
    logger.debug(f'columns list:{df.columns}')
    for key, value in agg_fun.items():
        #logger.debug(f'{key}:{df[key].dtype}')
        if key not in df:
            logger.debug(f"Can not find column:{key} in df:{file}")

    gp = df.groupby('card_id').agg(agg_fun)
    return gp


def get_summary_card_his_new():
    agg_fun = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean'],
        'category_2_2.0': ['mean'],
        'category_2_3.0': ['mean'],
        'category_2_4.0': ['mean'],
        'category_2_5.0': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'city_id': ['nunique'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'merchant_category_id': ['nunique'],
        'merchant_id': ['nunique'],
        'month_diff': ['mean'],
        'month_lag': ['mean', 'max', 'min', 'std'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'card_id':['count'],
    }

    his_df = _get_transaction(trans_his_file)
    history = _summary_card_trans_col(his_df, agg_fun)
    history.columns = [f'his_{"_".join(col)}' for col in history.columns]

    auth_df = his_df[his_df.authorized_flag==1]
    auth = _summary_card_trans_col(auth_df, agg_fun)
    auth.columns = [f'auth_{"_".join(col)}' for col in auth.columns]

    new_df = _get_transaction(trans_his_file)
    new = _summary_card_trans_col(new_df, agg_fun)
    new.columns = [f'new_{"_".join(col)}' for col in new.columns]

    df = pd.concat([history, auth, new], axis=1)

    for col in [col for col in df.columns if df[col].dtype.name == 'datetime64[ns]']:
        df[col] = pd.DatetimeIndex(df[col]).astype(np.int64) * 1e-9

    for col in [col for col in df.columns if df[col].dtype.name == 'timedelta64[ns]']:
        df[col] = df[col].dt.days

    return df


def get_month_trend_his_new():
    his = _aggregate_per_month(trans_his_file,'his')
    new = _aggregate_per_month(trans_new_file, 'new')
    return pd.concat([his,new], axis=1)

@timed()
def _aggregate_per_month(file, prefix):
    history = _get_transaction(file)
    grouped = history.groupby(['card_id', 'month_lag'])
    history['installments'] = history['installments'].astype(int)
    agg_func = {
        'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
        'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
    }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = [f"{prefix}_{'_'.join(col)}" for col in final_group.columns]
    #final_group.reset_index(inplace=True)

    return final_group

@timed()
def _successive_aggregates(file, field1, field2):
    if isinstance(file, pd.DataFrame):
        df = file
    else:
        df = _get_transaction(file)

    t = df.groupby(['card_id', field1])[field2].mean()
    u = pd.DataFrame(t).reset_index().groupby('card_id')[field2].agg(['mean', 'min', 'max', 'std'])
    u.columns = [field1 + '_' + field2 + '_' + col for col in u.columns.values]
    u.reset_index(inplace=True)
    return u

def get_summary_feature_agg():
    trans = _get_transaction(trans_new_file)
    additional_fields = _successive_aggregates(trans, 'category_1', 'purchase_amount')
    additional_fields = additional_fields.merge(_successive_aggregates(trans, 'installments', 'purchase_amount'),
                                                on='card_id', how='left')
    additional_fields = additional_fields.merge(_successive_aggregates(trans, 'city_id', 'purchase_amount'),
                                                on='card_id', how='left')
    additional_fields = additional_fields.merge(_successive_aggregates(trans, 'category_1', 'installments'),
                                                on='card_id', how='left')
    additional_fields.set_index('card_id', inplace=True)
    return additional_fields


@reduce_mem()
def get_feature_target():
    train = get_train_test(train_file)

    target = train.loc[:,'target']

    train = train.drop(columns=['target'])

    test = get_train_test(test_file)

    feature = pd.concat([train, test])

    his_auth_new = get_summary_card_his_new()

    summary_feature = get_summary_feature_agg()

    month_trend = get_month_trend_his_new()

    feature = pd.concat([feature, his_auth_new, summary_feature, month_trend], axis=1)

    feature.fillna(0, inplace=True)
    return feature.loc[target.index], target, feature.loc[ ~ (feature.index.isin( target.index))]



if __name__ == '__main__':

    pass
    #logger.debug(get_train().shape)



