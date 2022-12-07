import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

class TargetPlot:
    def __init__(self, sample, feature:str, target:str, type_target:str='bin', 
                bins: int=10, left_quant: float=-1.0, right_quant: float=-1.0, spec_value:float=False, log: bool=False):

        self.sample = sample
        self.feature = feature
        self.target = target
        self.bins = bins
        self.left_quant = left_quant
        self.right_quant = right_quant
        self.spec_value = spec_value
        self.log = log
        self.type_target = type_target

    def _bin_prep(self, df):
        df1 = df.groupby(self.feature).agg(
                {f'{self.target}': np.sum, 
                'index': np.size}).reset_index()

        df1.columns = [self.feature, 'target_count', 'all_count']
        df1['target_rate'] = 100*df1['target_count']/df1['all_count']
        df1['rolling_mean'] = np.nan
        return df1

    def _reg_prep(self, df):
        df1 = df.groupby(self.feature).agg(
                {f'{self.target}': np.mean, 
                'index': np.size}).reset_index()

        df1.columns = [self.feature, 'mean_target', 'all_count']
        return df1

    def get_bin_table(self):
        
        df = self.sample[[self.feature, self.target]]

        df_null = df[df[self.feature].isna()].reset_index(drop=True).reset_index()
        df_null[self.feature] = df_null[self.feature].astype(str)

        if self.type_target == 'bin':
            df_null1 = self._bin_prep(df_null)
        elif self.type_target == 'reg':
            df_null1 = self._reg_prep(df_null)
        
        # 1. очистили от пропусков
        df = df[df[self.feature].notna()].reset_index(drop=True)
        
        if self.spec_value is not False:
            spec = df[df[self.feature] == self.spec_value].reset_index(drop=True).reset_index()
            spec[self.feature] = 'spec: = '+str(self.spec_value)

            if self.type_target == 'bin':
                spec1 = self._bin_prep(spec)
            elif self.type_target == 'reg':
                spec1 = self._reg_prep(spec)

            # 2. Отфильтровали по квартилю
            df = df[df[self.feature] != self.spec_value].reset_index(drop=True)

        if self.log is True:
            df[self.feature] = np.log(df[self.feature]).reset_index(drop=True)

        # считаем квартили без пропусков
        if self.left_quant != -1.0:
            quant_l = round(df[self.feature].quantile(self.left_quant), 4)
        if self.right_quant != -1.0:
            quant_r = round(df[self.feature].quantile(1-self.right_quant), 4)

        if self.left_quant != -1.0:
            left = df[df[self.feature] < quant_l].reset_index(drop=True).reset_index()
            left[self.feature] = 'lq: < '+str(quant_l)

            if self.type_target == 'bin':
                left1 = self._bin_prep(left)
            elif self.type_target == 'reg':
                left1 = self._reg_prep(left)
            
            # 3. Отфильтровали по квартилю
            df = df[df[self.feature] >= quant_l].reset_index(drop=True)

        if self.right_quant != -1.0:
            right = df[df[self.feature] > quant_r].reset_index(drop=True).reset_index()
            right[self.feature] = 'rq: > '+str(quant_r)

            if self.type_target == 'bin':
                right1 = self._bin_prep(right)
            elif self.type_target == 'reg':
                right1 = self._reg_prep(right)
            
            # 4. Отфильтровали по квартилю
            df = df[df[self.feature] <= quant_r].reset_index(drop=True)
            
        df = df.reset_index(drop=True).reset_index()
        df[self.feature] = pd.cut(df[self.feature], self.bins)

        if self.type_target == 'bin':
            df1 = self._bin_prep(df)
            df1['rolling_mean'] = df1['target_rate'].rolling(window=3,
                                                        center=True,
                                                        win_type='triang').mean()
            number_1=len(df1)-1
            number_2=len(df1)-2

            df1.loc[0, 'rolling_mean'] = (df1.loc[0, 'target_rate'] + 
                                          df1.loc[1, 'target_rate']/2)
            df1.loc[number_1, 'rolling_mean'] = (df1.loc[number_1, 'target_rate'] + 
                                                 df1.loc[number_2, 'target_rate']/2)
        elif self.type_target == 'reg':
                df1 = self._reg_prep(df)
        
        if self.left_quant != -1.0 and self.right_quant != -1.0:
            if self.spec_value is False:
                final = pd.concat([left1, df1, right1, df_null1], axis=0).reset_index(drop=True)
            else:
                final = pd.concat([left1, df1, right1, df_null1, spec1], axis=0).reset_index(drop=True)
        elif self.left_quant != -1.0:
            if self.spec_value is False:
                final = pd.concat([left1, df1, df_null1], axis=0).reset_index(drop=True)
            else:
                final = pd.concat([left1, df1, df_null1, spec1], axis=0).reset_index(drop=True)
        elif self.right_quant != -1.0:
            if self.spec_value is False:
                final = pd.concat([df1, right1, df_null1], axis=0).reset_index(drop=True)
            else:
                final = pd.concat([df1, right1, df_null1, spec1], axis=0).reset_index(drop=True)
        else:
            if self.spec_value is False:
                final = pd.concat([df1, df_null1], axis=0).reset_index(drop=True)
            else:
                final = pd.concat([df1, df_null1, spec1], axis=0).reset_index(drop=True)
        final[self.feature] = final[self.feature].astype(str)
        return final.reset_index(drop=True).reset_index()


    def _target_plot(self, bin_df):
    
        # Подготовка данных для построения графика
        plt.rcParams['figure.figsize'] = (10, 5)
        fig, ax = plt.subplots()
        ax2 = ax.twinx() 

        ax.bar(
            bin_df[self.feature],
            bin_df["all_count"],
            color='#cda0aa',
            label=f'{self.feature}')

        if self.type_target == 'bin':
            ax.bar(
                bin_df[self.feature],
                bin_df["target_count"],
                color='red',
                label='target_count')

            ax2.plot(
                bin_df[self.feature],
                bin_df['rolling_mean'],
                color=(24/254, 192/254, 196/254),
                label='target - скользящее среднее')

            ax2.plot(
                bin_df[self.feature],
                bin_df['target_count'],
                color=(246/254, 115/254, 109/254),
                label='target - оригинальный')

        elif self.type_target == 'reg':
            ax2.plot(
                bin_df[self.feature],
                bin_df['mean_target'],
                color=(246/254, 115/254, 109/254),
                label='target - оригинальный')
        
        
        #ax.set_xticklabels(bin_df[feature])
        ax.set_xticklabels(bin_df['index'])
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        #  Настраиваем вид основных тиков:
        ax.tick_params(
            axis = 'both',    #  Применяем параметры к обеим осям
            which = 'major',    #  Применяем параметры к основным делениям
            direction = 'inout',    #  Рисуем деления внутри и снаружи графика
            length = 20,    #  Длина делений
            width = 2,     #  Ширина делений
            color = 'black',    #  Цвет делений
            pad = 10,    #  Расстояние между черточкой и ее подписью
            labelsize = 10,    #  Размер подписи
            labelcolor = 'black',    #  Цвет подписи
            bottom = True,    #  Рисуем метки снизу
            #top = True,    #   сверху
            left = True,    #  слева
            #right = True,    #  и справа
            labelbottom = True,    #  Рисуем подписи снизу
            #labeltop = True,    #  сверху
            labelleft = True,    #  слева
            #labelright = True,    #  и справа
            labelrotation = 0)    #  Поворот подписей

        ax.set_ylabel('Кол-во записей', fontsize=12)

        if self.type_target == 'bin':
            ax2.set_ylabel('Доля таргета, %', fontsize=12)
        elif self.type_target == 'reg':
            ax2.set_ylabel('Средний таргет', fontsize=12)

        if self.left_quant != -1.0 and self.right_quant != -1.0:
            plt.title(f'Отсечение по {100*self.left_quant} % и {100*(1-self.right_quant)} % персентилю', fontsize= 13)
        elif self.left_quant != -1.0:
            plt.title(f'Отсечение по {100*self.left_quant} % персентилю', fontsize= 13)
        elif self.right_quant != -1.0:
            plt.title(f'Отсечение по {100*(1-self.right_quant)} % персентилю', fontsize= 13)
        else:
            plt.title('Без отсечения', fontsize= 13)

        plt.grid(True)
        plt.tight_layout()

    def get_target_plot(self):
        print(f'==================== Отрисовка для фактора {self.feature} ====================')
        bin_df = self.get_bin_table()
        display(bin_df)
        self._target_plot(bin_df)