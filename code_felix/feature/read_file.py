from code_felix.utils_.util_pandas import *
from code_felix.utils_.util_cache_file import *
from functools import lru_cache
trans_new_file = './input/new_merchant_transactions.csv'

trans_his_file = './input/historical_transactions.csv'

merchants_file = './input/merchants.csv'

train_file = './input/train.csv'

test_file = './input/test.csv'

def _get_transaction(file):

    trans =  pd.read_csv(file, parse_dates=['purchase_date'])

    merchants = pd.read_csv(merchants_file)

    merchants.columns = [ f'mer_{col}' if col != 'merchant_id' else col for col in merchants.columns ]

    return  pd.merge(trans, merchants, how='left', on='merchant_id' )



max_date = pd.to_datetime('2018-02-01')

def get_train_test(file):
    df =  pd.read_csv(file, parse_dates=['first_active_month'])
    #df.first_active_month.fillna(pd.to_datetime('2017-03-09'), inplace=True)
    df.loc[df.card_id == 'C_ID_c27b4f80f7', 'first_active_month'] = pd.to_datetime('2017-03-09')
    date_parts = ["year", "weekday", "month"]
    for part in date_parts:
        part_col = 'first_active_month' + "_" + part
        df[part_col] = getattr(df['first_active_month'].dt, part).astype(int)
    df['elapsed_time'] = (max_date - df['first_active_month']).dt.days

    return df




def _summary_trans_col(file, agg_fun):
    df = _get_transaction(file)

    df['authorized_flag'] = df['authorized_flag'].apply(lambda x: 1 if x == 'Y' else 0)

    df['purchase_month']  = df['purchase_date'].dt.month

    map_dict = {'Y': 0, 'N': 1}
    df['category_1'] = df['category_1'].apply(lambda x: map_dict[x])

    map_dict = {'A': 0, 'B': 1, 'C': 2, 'nan': 3, np.nan: 3}
    df['category_3'] = df['category_3'].apply(lambda x: map_dict[x])

    df = pd.get_dummies(df, columns=['category_2', 'category_3'])
    logger.debug(f'columns list:{df.columns}')
    for key, value in agg_fun.items():
        logger.debug(f'{key}:{df[key].dtype}')
        if key not in df:
            logger.debug(f"Can not find column:{key} in df:{file}")

    gp = df.groupby('card_id').agg(agg_fun)
    return gp


def get_summary_his_new():
    agg_fun = {
        'authorized_flag': ['sum', 'mean'],
        'category_1': ['sum', 'mean'],
        'category_2_1.0': ['mean', 'sum'],
        'category_2_2.0': ['mean', 'sum'],
        'category_2_3.0': ['mean', 'sum'],
        'category_2_4.0': ['mean', 'sum'],
        'category_2_5.0': ['mean', 'sum'],
        'category_3_1': ['sum', 'mean'],
        'category_3_2': ['sum', 'mean'],
        'category_3_3': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'max', 'min'],
        'month_lag': ['min', 'max'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'city_id': ['nunique'],
    }

    history = _summary_trans_col(trans_his_file, agg_fun)
    history.columns = [f'his_{"_".join(col)}' for col in history.columns]

    new = _summary_trans_col(trans_new_file, agg_fun)
    new.columns = [f'new_{"_".join(col)}' for col in new.columns]
    df = pd.concat([history, new], axis=1)

    for col in [col for col in df.columns if df[col].dtype.name == 'datetime64[ns]']:
        df[col] = pd.DatetimeIndex(df[col]).astype(np.int64) * 1e-9

    for col in [col for col in df.columns if df[col].dtype.name == 'timedelta64[ns]']:
        df[col] = df[col].dt.days

    return df


def get_month_trend_his_new():
    his = _aggregate_per_month(trans_his_file,'his')
    new = _aggregate_per_month(trans_new_file, 'new')
    return pd.concat([his,new], axis=1)


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

if __name__ == '__main__':

    pass
    #logger.debug(get_train().shape)



