import pandas as pd

def get_uplift(data:pd.DataFrame, target:str='target', treatment:str='treatment'):

    kk = data.groupby([target, treatment]).size().reset_index()
    kk.columns = [target, treatment, 'count']
    kk = kk.sort_values([treatment])

    tr_bad1 = int(kk['count'][(kk[treatment] == 0) & (kk[target] == 0)])
    tr_bad2 = int(kk['count'][(kk[treatment] == 0) & (kk[target] == 1)])

    tr_good1 = int(kk['count'][(kk[treatment] == 1) & (kk[target] == 0)])
    tr_good2 = int(kk['count'][(kk[treatment] == 1) & (kk[target] == 1)])

    rate_ctr = tr_bad2/(tr_bad1+tr_bad2)
    rate_tr = tr_good2/(tr_good1+tr_good2)

    print('treatment_rate: ', round(rate_tr, 5))
    print('control_rate: ', round(rate_ctr, 5))
    print('absolute_uplift: ', round(rate_tr-rate_ctr, 5))

    return kk