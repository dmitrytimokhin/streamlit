import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class BaseEda:
    def __init__(self, df):
        """[summary]

        Args:
            df ([type]): [description]
        """
        self.df = df.copy()
    
    def base_overview(self):
        #print(f'Количество строк: {self.df.shape[0]}')
        #print(f'Количество факторов: {self.df.shape[1]}')
        #print()

        df_type = self._df_types()
        df_null = self._df_null()
        df_desc = self._describe()
        
        df_unique = pd.DataFrame(self.df.nunique()).reset_index()
        df_unique.columns = ['feature', 'count_unique']
        
        df_all = df_type.merge(
            df_null,
            left_on='feature',
            right_on='feature',
            how='left').merge(
                        df_unique,
                        left_on='feature',
                        right_on='feature',
                        how='left')
        
        df_all = df_all.merge(
            df_desc,
            left_on='feature',
            right_on='feature',
            how='left'
            )
        
        return df_all

    def _df_types(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        df_types = pd.DataFrame(self.df.dtypes).reset_index()
        df_types.columns = ['feature', 'type']
        return df_types
        
    def _df_null(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        df_null = pd.DataFrame(self.df.isna().sum()).reset_index()
        df_null.columns = ['feature', 'count_null']
        df_null['per_null'] = round(df_null['count_null']*100/self.df.shape[0], 3)
        return df_null
        
    def _describe(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        df_desc = self.df.describe().T.reset_index()
        df_desc = df_desc.rename(columns={'index':'feature'})
        
        df_sk = pd.DataFrame(self.df.skew()).reset_index()
        df_sk.columns = ['feature', 'skew']
        
        df_kur = pd.DataFrame(self.df.kurtosis()).reset_index()
        df_kur.columns = ['feature', 'kurtosis']
        
        df_all = df_desc.merge(
            df_sk,
            left_on='feature',
            right_on='feature',
            how='left').merge(
                        df_kur,
                        left_on='feature',
                        right_on='feature',
                        how='left')
        return df_all
    
    def visual_null(self, path: str='-1'):
        plt.figure(figsize = (16, 7))

        plt.subplot(1,2,1)
        sns.heatmap(self.df.isnull(), cbar=False)
        plt.xticks(rotation = 35,
                   horizontalalignment='right',
                   fontweight='light'  )
        plt.title('Dataset missing values')

        if path != '-1':
            plt.savefig(f'{path}/visual_null.png')
            plt.close()
        
    def cat_plot(self, column: str, path: str='-1'):
        """Построение графика одного категориального фактора

        Args:
            column (str): [description]
        """
        #dict_unique_values = dict(self.df.nunique())
        
        # set size of the plot
        plt.figure(figsize=(10, 6)) 

        # countplot shows the counts of observations in each categorical bin using bars.
        # x - name of the categorical variable
        ax = sns.countplot(x = column, data = self.df, palette=["#eed4d0", "#a2708e"])

        # set the current tick locations and labels of the x-axis.
        plt.xticks(np.arange(len(list(self.df[column].unique()))), list(self.df[column].unique()))
        # set title
        plt.title(f'Кол-во записей для каждой категории {column}',fontsize= 14)
        # set x label
        plt.xlabel('Категории')
        # set y label
        plt.ylabel('Количество записей')

        # calculate passengers for each category
        labels = (self.df[column].value_counts())
        
        # add result numbers on barchart
        for i, v in enumerate(labels):
            if v == labels.max() or v >= labels.max()*0.7:
                ax.text(i, v/2, str(v), 
                        horizontalalignment = 'center', 
                        size = 14, 
                        color = '#3f3e6fd1', 
                        fontweight = 'medium')
            else:
                ax.text(i, v + 0.05*labels.max(), str(v), 
                        horizontalalignment = 'center', 
                        size = 14, 
                        color = '#3f3e6fd1', 
                        fontweight = 'medium')

        plt.tight_layout()

        if path != '-1':
            plt.savefig(f'{path}/{column}_cat_plot.png')
            plt.close()
    
    def num_plot(self, column: str, path: str='-1'):
        """Построение графика одного непрерывного фактора

        Args:
            column (str): название непрерывного фактора
        """
        fig, axs = plt.subplots(2, figsize=(12, 12))
        fig.suptitle(f'BoxPlot and StripPlot for {column}',fontsize= 16)

        palette = sns.cubehelix_palette(5, start = 3)

        sns.boxplot(x = column, 
                    data = self.df,
                    palette = palette, 
                    fliersize = 0, ax=axs[0])

        sns.stripplot(x = column, 
                      data = self.df,
                      linewidth = 0.6, 
                      palette = palette, ax=axs[0])

        plt.xlabel('Age')

        age_1_class = self.df[(self.df[column].notna())]
        sns.kdeplot(age_1_class[column], shade=True, color='#a2708e', ax=axs[1])

        plt.title(f'KdePlot for {column}',fontsize= 16)
        plt.xlabel(f'{column}',fontsize= 16)
        plt.tight_layout()
        #plt.show()

        if path != '-1':
            plt.savefig(f'{path}/{column}_num_plot.png')
            plt.close()

    def cut_num_plot(self, column1: str, column2: str):
        """Построение графика категориального фактора против непрерывного фактора

        Args:
            column1 (str): название категориального фактора
            column2 (str): название непрерывного фактора
        """
        
        dict_unique_values = dict(self.df.nunique())
        
        # set size
        plt.figure(figsize=(30, 10))

        # set palette
        palette = sns.cubehelix_palette(5, start = 3)

        plt.subplot(1, 2, 1)
        sns.boxplot(x = column1, 
                    y = column2, 
                    data = self.df,
                    palette = palette, 
                    fliersize = 0)

        sns.stripplot(x = column1, 
                      y = column2, 
                      data = self.df,
                      linewidth = 0.6, 
                      palette = palette)

        plt.xticks(np.arange(dict_unique_values[column1]), list(self.df[column1].unique()))
        plt.title(f'{column2} distribution grouped by {column1}',fontsize= 16)
        plt.xlabel(f'{column1}')

        plt.show()
        
    def num_target_plot(self, column: str, target: str, path: str='-1'):
        """Построение непрерывного фактора против таргета

        Args:
            column (str): название непрерывного фактора
            target (str): название категориального фактора
        """
        plt.figure(figsize=(20, 6))
        #palette = sns.cubehelix_palette(5, start = 3)
        plt.subplot(1, 2, 1)

        age_0_class = self.df[(self.df[column].notna()) & 
                         (self.df[target] == 0)]
        age_1_class = self.df[(self.df[column].notna()) & 
                         (self.df[target] == 1)]

        sns.kdeplot(age_0_class[column], shade=True, color='#eed4d0', label = '0 target')
        sns.kdeplot(age_1_class[column], shade=True,  color='#cda0aa', label = '1 target')

        plt.title(f'{column} distribution grouped by target',fontsize= 16)
        plt.xlabel(column)
        plt.xlim(0, 90)
        plt.tight_layout()
        plt.legend()

        if path != '-1':
            plt.savefig(f'{path}/{column}_num_target_plot.png')
            plt.close()

    def num_target_plot_advanced(self, name_feature: str, name_target: str, bins: int = 20, quant: float=-1.0, path: str='-1'):
        """
        Построение графика - распределение таргета по бинам

        Args:
            name_feature (str): Название фактора
            name_target (str): Название таргета
            bins (int): Кол-во бинов для разбиения
            quant (float): Процент отрезания по квантилям (например, 0.05). Если -1, то нет отрезания
            path ([type]): Путь для сохранения. Если '-1', то график рисуется в источнике
        """
    
        # Подготовка данных для построения графика
        sample_copy = self.df.copy()
        sample_copy = sample_copy.reset_index()

        sample_copy = sample_copy[[name_target, 'index', name_feature]]
    
        if quant != -1.0:
            sample_copy = sample_copy[((sample_copy[name_feature] >= sample_copy[name_feature].quantile(quant))&
                                       (sample_copy[name_feature] <= sample_copy[name_feature].quantile(1-quant)))]
    
        # Пробегаемся по не нулевым факторам
        sample_copy[name_feature] = pd.cut(sample_copy[name_feature], bins)
    
        sample_copy_new = sample_copy.groupby(name_feature).agg(
                                      {f'{name_target}': np.sum, 
                                        'index': np.size}).reset_index()
    
        sample_copy_new[name_feature] = sample_copy_new[name_feature].astype(str)
    
        # Пробегаемся по нулевым факторам
        sample_copy1 = sample_copy[sample_copy[name_feature].isna()]
    
        if sample_copy1.shape[0] > 0:
            sample_copy1[name_feature] = sample_copy1[name_feature].astype(str)
            sample_copy1[name_feature] = sample_copy1[name_feature].fillna('None')
        
            sample_copy_new1 = sample_copy1.groupby(name_feature).agg(
                                            {f'{name_target}': np.sum, 
                                              'index': np.size}).reset_index()
        
            sample_copy_new = pd.concat([sample_copy_new, sample_copy_new1], axis=0).reset_index(drop=True)
    
    
        sample_copy_new.columns = [f'{name_feature}', 'count_target', 'count_feature']
        sample_copy_new['percent_target'] = sample_copy_new['count_target']/sample_copy_new['count_feature']

        sample_copy_new['rolling_mean'] = sample_copy_new['percent_target'].rolling(window=3,
                                                                                    center=True,
                                                                                    win_type='triang').mean()

        number_1=len(sample_copy_new)-1
        number_2=len(sample_copy_new)-2

        sample_copy_new.loc[0, 'rolling_mean'] = (sample_copy_new.loc[0, 'percent_target'] + 
                                                  sample_copy_new.loc[1, 'percent_target']/2)
        sample_copy_new.loc[number_1, 'rolling_mean'] = (sample_copy_new.loc[number_1, 'percent_target'] + 
                                                         sample_copy_new.loc[number_2, 'percent_target']/2)


        # Подготовка данных для построения графика
        plt.rcParams['figure.figsize'] = (10, 5)
        fig, ax = plt.subplots()
        ax2 = ax.twinx() 

        ax.bar(sample_copy_new[name_feature],
               sample_copy_new["count_feature"],
               color='#cda0aa',
               label=f'{name_feature}')
    
        ax2.plot(sample_copy_new[name_feature],
                 sample_copy_new['rolling_mean'],
                 color=(24/254, 192/254, 196/254),
                 label='target - скользящее среднее')
    
        ax2.plot(sample_copy_new[name_feature],
                 sample_copy_new['percent_target'],
                 color=(246/254, 115/254, 109/254),
                 label='target - оригинальный')

        ax.set_xticklabels(sample_copy_new[name_feature])
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
            labelrotation = 90)    #  Поворот подписей
    
        ax.set_ylabel('Кол-во записей', fontsize=12)
        ax2.set_ylabel('Доля таргета, %', fontsize=12)

        if quant != -1.0:
            plt.title(f'Отсечение по {100*quant} % персентилю', fontsize= 13)
        else:
            plt.title('Без отсечения', fontsize= 13)

        plt.grid(True)
        plt.tight_layout()
    
        if path != '-1':
            plt.savefig(f'{path}/{name_feature}{quant}_ft.png')
            plt.close()
        
        
        