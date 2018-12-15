import pandas as pd
trans_his = pd.read_csv('./input/historical_transactions.csv')
trans_new = pd.read_csv('./input/new_merchant_transactions.csv')

sample_count = 30000
card_id_sample = trans_his.card_id.drop_duplicates().sample(sample_count)

trans_his = trans_his[trans_his.card_id.isin(card_id_sample)]
trans_his.to_csv(f'./input/historical_transactions{sample_count}.csv', index=None)


trans_new = trans_new[trans_new.card_id.isin(card_id_sample)]
trans_new.to_csv(f'./input/new_merchant_transactions{sample_count}.csv', index=None)


