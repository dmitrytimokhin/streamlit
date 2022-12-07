import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt

class WingUplift:
    
    def __init__(self, df, target, treatment, score, strategy, woe, ascending=False):
        
        self.df = df
        self.target = target
        self.treatment = treatment
        self.score = score
        self.strategy = strategy
        self.woe = woe
        self.ascending = ascending
        
    def build_table(self):
        
        self.df = self.df[[self.target, self.treatment, self.score]]
        
        if self.strategy == 'all':
            df1 = self.df.copy()
        elif self.strategy == 'trt':
            df1 = self.df[self.df[self.treatment] == 1]
        elif self.strategy == 'ctrl':
            df1 = self.df[self.df[self.treatment] == 0]
        
        # учимся на части выборки
        self.woe.fit(df1[[self.score]], df1[[self.target]])
        
        # преобразовываем всю изначальную выборку
        new_df1 = self.woe.transform(self.df[[self.score]])
        
        df1 = pd.concat([self.df, new_df1], axis=1) 
        df1['woe_group'] = df1['woe_group'].astype(int)

        self.woe_table = self.woe.get_wing_agg().reset_index()
        
        kk = self.woe_table.copy()
        kk = kk[['grp', 'woe', 'local_event_rate', 'bin']]
        kk.columns = ['woe_group', 'woe', 'local_event_rate', 'bin']
        kk['woe_group'] = kk['woe_group'].astype(int)

        df1 = df1.merge(
            kk,
            left_on = 'woe_group',
            right_on = 'woe_group',
            how = 'left'
        )

        final = ps.sqldf(
            f'''
            WITH trt AS (
                SELECT bin, local_event_rate, woe_group, 
                    count(*) AS n_trt, SUM({self.target}) AS tar1_trt, 
                    count({self.target})-sum({self.target}) AS tar0_trt, AVG({self.score}) AS mean_pred_trt
                FROM df1
                WHERE treatment = 1
                GROUP BY bin, local_event_rate, woe_group
                ORDER BY woe_group
            ),

            ctrl AS (
                SELECT bin, local_event_rate, woe_group, 
                    count(*) AS n_ctrl, SUM({self.target}) AS tar1_ctrl, 
                    COUNT({self.target})-SUM({self.target}) AS tar0_ctrl, AVG({self.score}) AS mean_pred_ctrl
                FROM df1
                WHERE treatment = 0
                GROUP BY bin
                ORDER BY woe_group
            ),
            
            all_trt AS (
                SELECT 'total' AS bin, count(*) AS n_trt, SUM({self.target}) AS tar1_trt, 
                    count({self.target})-sum({self.target}) AS tar0_trt
                FROM df1
                WHERE treatment = 1

            ),
            
            all_ctrl AS (
                SELECT 'total' AS bin, count(*) AS n_ctrl, SUM({self.target}) AS tar1_ctrl, 
                    COUNT({self.target})-SUM({self.target}) AS tar0_ctrl, AVG({self.score}) AS mean_pred_ctrl
                FROM df1
                WHERE treatment = 0
            ),
            
            all_t AS (
                SELECT 'total' AS bin, 'total' AS local_event_rate, 'total' AS woe_group, 
                    all_trt.n_trt, all_trt.tar1_trt, all_trt.tar0_trt, 
                    all_ctrl.n_ctrl, all_ctrl.tar1_ctrl, all_ctrl.tar0_ctrl
                FROM all_trt
                LEFT JOIN all_ctrl
                    ON all_trt.bin = all_ctrl.bin
            )

            SELECT trt.bin, trt.local_event_rate, trt.woe_group, 
                trt.n_trt, trt.tar1_trt, trt.tar0_trt, 
                ctrl.n_ctrl, ctrl.tar1_ctrl, ctrl.tar0_ctrl
            FROM trt
            LEFT JOIN ctrl
                ON trt.bin = ctrl.bin
                AND trt.woe_group = ctrl.woe_group
                
            UNION
            
            SELECT * 
            FROM all_t
        '''
        )

        final['resp_rate_trt'] = final['tar1_trt']/final['n_trt']
        final['resp_rate_ctrl'] = final['tar1_ctrl']/final['n_ctrl']
        final['real_uplift'] = final['resp_rate_trt'] - final['resp_rate_ctrl']
        
        sort = final[final['woe_group'] != 'total'].sort_values(['woe_group'], ascending=self.ascending).reset_index(drop=True)
        total = final[final['woe_group'] == 'total'].reset_index(drop=True)
        
        final = pd.concat([sort, total], axis=0).reset_index(drop=True)
        
        self.final = final
        
    def plot_table(self):
        
        self.final = self.final[self.final['woe_group'] != 'total']
        self.final = self.final.sort_values(['woe_group'], ascending=self.ascending).reset_index(drop=True)
        self.final = self.final.reset_index()
        
        percentiles = self.final['index']
        response_rate_trmnt = self.final['resp_rate_trt']
        response_rate_ctrl = self.final['resp_rate_ctrl']
        uplift_score = self.final['real_uplift']

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