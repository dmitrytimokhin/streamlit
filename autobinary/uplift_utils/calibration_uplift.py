import pandas as pd
import pandasql as ps
import numpy as np

import matplotlib.pyplot as plt

class UpliftCalibration:
    
    def __init__(self, df:pd.DataFrame, type_score: str='probability', type_calib: str='bins', 
                 strategy: str='all', woe: object=None, bins: int=5):
        
        self.df = df ## Датафрейм с таргетом, флагом коммуникации и скорром
        self.type_score = type_score ## Тип скорра для калибровки ('probability' / 'uplift')
        self.type_calib = type_calib ## Тип калибровки, через перцентиль или с помощью woe ('bins' / 'woe')
        self.strategy = strategy ## Вся выборка ('all') / С коммуникацией ('trt') / Без коммуникации ('crtl')
        self.woe = woe ## Объект обучения для WOE
        self.bins = bins ## Количество бинов для перцентиля
        
    def fit(self,target:str='target',treatment:str='treatment',score:str='proba',ascending:bool=False):
        
        ## target - название столбца таргета
        ## treatment - название столбца флага коммуникации
        ## score - название столбца со скорром
        ## ascending - направление калибровки

        # учимся на части выборки    
            
        if self.type_calib=='bins':
            df1 = self.df.sort_values(score, ascending=False).reset_index(drop=True)

            # возвращается кортеж:
            _, self.list_bounders = pd.qcut(
                df1[score],
                q=self.bins,
                precision=10,
                retbins=True) 

            percentiles1 = [round(p * 100 / len(self.list_bounders)) for p in range(2, len(self.list_bounders)+1)]

            percentiles = [f"0-{percentiles1[0]}"] + \
            [f"{percentiles1[i]}-{percentiles1[i + 1]}" for i in range(len(percentiles1)-1)]

            if self.type_score == 'uplift':
                self.list_bounders[0] = -100
                self.list_bounders[len(self.list_bounders)-1] = 100
            else:
                self.list_bounders[0] = 0
                self.list_bounders[len(self.list_bounders)-1] = 1            
        
        else:
            if self.strategy == 'all':
                df1 = self.df
            elif self.strategy == 'trt':
                df1 = self.df[self.df[treatment] == 1]
            elif self.strategy == 'ctrl':
                df1 = self.df[self.df[treatment] == 0]
               
            self.woe.fit(df1[[score]], df1[[target]])

            # преобразовываем всю изначальную выборку
            new_df1 = self.woe.transform(self.df[[score]])
            df1 = pd.concat([self.df, new_df1], axis=1)

            self.list_bounders = self.woe.optimal_edges.tolist()
            if self.type_score == 'uplift':
                self.list_bounders[0] = -100
                self.list_bounders[len(self.list_bounders)-1] = 100
            else:
                self.list_bounders[0] = 0
                self.list_bounders[len(self.list_bounders)-1] = 1  

            percentiles1 = [round(p * 100 / len(self.list_bounders)) for p in range(2, len(self.list_bounders)+1)]

            percentiles = [f"0-{percentiles1[0]}"] + \
                [f"{percentiles1[i]}-{percentiles1[i + 1]}" for i in range(len(percentiles1)-1)]

        percentiles.reverse()


        df1['interval'] = pd.cut(df1[score], bins=self.list_bounders, precision=10)
        df1['name_interval'] = pd.cut(df1[score], bins=self.list_bounders, labels=percentiles)

        df1['left_b'] = df1['interval'].apply(lambda x: x.left)
        df1['right_b'] = df1['interval'].apply(lambda x: x.right)
        
        df1['interval'] = df1['interval'].astype(str)
        final = ps.sqldf(
            f'''
            WITH trt AS (
                SELECT interval, name_interval, left_b, right_b,
                    count(*) AS n_trt, SUM(target) AS tar1_trt, 
                    count({target})-sum({target}) AS tar0_trt, AVG({score}) AS mean_pred_trt
                FROM df1
                WHERE treatment = 1
                GROUP BY interval, name_interval, left_b, right_b
                ORDER BY interval
            ),

            ctrl AS (
                SELECT interval, name_interval, left_b, right_b, 
                    count(*) AS n_ctrl, SUM({target}) AS tar1_ctrl, 
                    COUNT({target})-SUM({target}) AS tar0_ctrl, AVG({score}) AS mean_pred_ctrl
                FROM df1
                WHERE treatment = 0
                GROUP BY interval, name_interval, left_b, right_b
                ORDER BY interval
            ),

            all_trt AS (
                SELECT 'total' AS interval, count(*) AS n_trt, SUM({target}) AS tar1_trt, 
                    count({target})-sum({target}) AS tar0_trt
                FROM df1
                WHERE treatment = 1

            ),

            all_ctrl AS (
                SELECT 'total' AS interval, count(*) AS n_ctrl, SUM({target}) AS tar1_ctrl, 
                    COUNT({target})-SUM({target}) AS tar0_ctrl, AVG({score}) AS mean_pred_ctrl
                FROM df1
                WHERE treatment = 0
            ),

            all_t AS (
                SELECT 'total' AS interval, 'total' AS name_interval, 'total' AS left_b, 'total' AS right_b, 
                    all_trt.n_trt, all_trt.tar1_trt, all_trt.tar0_trt, 
                    all_ctrl.n_ctrl, all_ctrl.tar1_ctrl, all_ctrl.tar0_ctrl
                FROM all_trt
                LEFT JOIN all_ctrl
                    ON all_trt.interval = all_ctrl.interval
            )

            SELECT trt.interval, trt.name_interval, trt.left_b, trt.right_b, 
                trt.n_trt, trt.tar1_trt, trt.tar0_trt, 
                ctrl.n_ctrl, ctrl.tar1_ctrl, ctrl.tar0_ctrl
            FROM trt
            LEFT JOIN ctrl
                ON trt.interval = ctrl.interval
                AND trt.name_interval = ctrl.name_interval

            UNION

            SELECT * 
            FROM all_t
        '''
        )
        
        final['resp_rate_trt'] = final['tar1_trt']/final['n_trt']
        final['resp_rate_ctrl'] = final['tar1_ctrl']/final['n_ctrl']
        final['real_uplift'] = final['resp_rate_trt'] - final['resp_rate_ctrl']

        self.df = None
        self.final = final.to_dict()
        
        return final
        
    def apply(self, score: pd.DataFrame, precision: int=10):
        
        df = pd.DataFrame({'score': score, 'interval': pd.cut(score, bins=self.list_bounders, precision=precision).astype(str)})
        df = df.merge(
            pd.DataFrame(self.final)[['interval', 'name_interval', 'real_uplift']],
            on = 'interval',
            how='left'
        )
        
        self.applied = df.to_dict()  
        
        return df
        
    def plot_table(self, ascending: bool=False):
        
        df = pd.DataFrame(self.final)
        df = df[df['interval'] != 'total']
        df = df.sort_values(['interval'], ascending=ascending).reset_index(drop=True)
        df = df.reset_index()
        
        percentiles = df['name_interval']
        response_rate_trmnt = df['resp_rate_trt']
        response_rate_ctrl = df['resp_rate_ctrl']
        uplift_score = df['real_uplift']

        _, axes = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        axes.errorbar(
            percentiles, 
            response_rate_trmnt, 
            linewidth=2, 
            color='forestgreen', 
            label='treatment\nresponse rate')

        axes.errorbar(
            percentiles, 
            response_rate_ctrl,
            linewidth=2, 
            color='orange', 
            label='control\nresponse rate')

        axes.errorbar(
            percentiles, 
            uplift_score, 
            linewidth=2, 
            color='red', 
            label='uplift')

        axes.fill_between(percentiles, response_rate_trmnt,
                          response_rate_ctrl, alpha=0.1, color='red')

        axes.legend(loc='upper right')
        axes.set_title(
            f'Uplift by percentile')
        axes.set_xlabel('Percentile')
        axes.set_ylabel(
            'Uplift = treatment response rate - control response rate')
        axes.grid()