import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

class BalanceCover:
    def __init__(self, entr_df, target='target'):
        self.entr_df = entr_df
        self.target=target

    def sample_describe(self):
        df = self.entr_df.copy()
        print('Всего записей в выборке: ', df.shape[0])
        print('Всего таргетов в выборке: ', df[df[self.target] == 1].shape[0])
    
    def _calc_balance(self, counter):
    
        df = self.entr_df.copy()
        
        # базовый баланс классов
        bal1 = df[df[self.target] == 1].shape[0]/df.shape[0]
        # баланс классов в бакете
        bal2 = df[0:counter][df[self.target] == 1].shape[0]/counter
        # количество таргетов в бакете
        count_t = df[0:counter][df[self.target] == 1].shape[0]
        
        return 100*bal1, 100*bal2, bal2/bal1, count_t, 100*df[0:counter][df[self.target] == 1].shape[0]/df[df[self.target] == 1].shape[0]

    def calc_scores(self, step, end):

        l_balans = []
        l_cover = []
        l_count = []
        l_base_bal = []
        l_bucket_bal = []
        l_turget_bucket = []

        for value in range(step, end+step, step):
            base_bal, bucket_bal, bal, turget_bucket, cov = self._calc_balance(counter=value)
            l_balans.append(bal)
            l_cover.append(cov)
            l_count.append(value)
            l_base_bal.append(base_bal)
            l_bucket_bal.append(bucket_bal)
            l_turget_bucket.append(turget_bucket)
    
        df_output = pd.DataFrame()
        
        df_output['start_bucket'] = l_count
        df_output['start_bucket'] = df_output['start_bucket'] - df_output['start_bucket']
        df_output['end_bucket'] = l_count
        df_output['turget_in_bucket'] = l_turget_bucket
        df_output['bucket_bal (%)'] = l_bucket_bal
        df_output['coverage (%)'] = l_cover
        df_output['base_bal (%)'] = l_base_bal
        df_output['bucket_bal/base_bal'] = l_balans
        
        self.output = df_output
        
    def plot_scores(self):
        plt.plot(self.output['end_bucket'], self.output['bucket_bal/base_bal'], label = 'Отношение балансов')
        plt.plot(self.output['end_bucket'], self.output['coverage (%)'], label = 'Покрытие')
        plt.grid()
        plt.legend()
        plt.title('Отношение балансов и покрытие')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('Процент(%)/выигрыш(раз)')
        plt.show()
        
    def _calc_balance_2(self, counter, step):
    
        df = self.entr_df.copy()
        
        # базовый баланс классов
        bal1 = df[df[self.target] == 1].shape[0]/df.shape[0]
        # баланс классов в бакете
        bal2 = df[counter:counter+step][df[self.target] == 1].shape[0]/(step)
        # количество таргетов в бакете
        count_t = df[counter:counter+step][df[self.target] == 1].shape[0]
        
        return 100*bal1, 100*bal2, bal2/bal1, count_t, 100*df[counter:counter+step][df[self.target] == 1].shape[0]/df[df[self.target] == 1].shape[0]
    
    def calc_scores_2(self, step, end):

        l_balans = []
        l_cover = []
        l_count = []
        l_base_bal = []
        l_bucket_bal = []
        l_turget_bucket = []

        for value in range(0, end, step):
            base_bal, bucket_bal, bal, turget_bucket, cov = self._calc_balance_2(counter=value, step=step)
        
            l_balans.append(bal)
            l_cover.append(cov)
            l_count.append(value)
            l_base_bal.append(base_bal)
            l_bucket_bal.append(bucket_bal)
            l_turget_bucket.append(turget_bucket)
    
        df_output2 = pd.DataFrame()
        
        df_output2['start_bucket'] = l_count
        df_output2['end_bucket'] = df_output2['start_bucket']+step
        df_output2['target_in_bucket'] = l_turget_bucket
        df_output2['bucket_bal (%)'] = l_bucket_bal
        df_output2['coverage (%)'] = l_cover
        df_output2['base_bal (%)'] = l_base_bal
        df_output2['bucket_bal/base_bal'] = l_balans
        
        self.output2 = df_output2
        
    def plot_scores_2(self):
        plt.plot(self.output2['start_bucket'], self.output2['bucket_bal (%)'])
        #plt.plot(self.output2['end_bucket'], self.output2['coverage (%)'], label = 'Покрытие')
        plt.grid()
        #plt.legend()
        plt.title('Процент таргетов в бакете')
        plt.xlabel('Кол-во клиентов')
        plt.ylabel('Процент(%)')
        plt.show()