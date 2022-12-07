import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class NumEda:
    def __init__(self, df):
        """[summary]

        Args:
            df ([type]): [description]
        """
        self.df = df.copy()
    
    def simple_plot(self, column: str, path: str='-1'):
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
        
    def target_plot(self, column: str, target: str, path: str='-1'):
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

    def advanced_target_plot(self, name_feature: str, name_target: str, bins: int = 20, quant: float=-1.0, path: str='-1'):
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