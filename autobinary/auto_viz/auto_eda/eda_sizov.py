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
        print(f'Колчество строк: {self.df.shape[0]}')
        print(f'Колчество факторов: {self.df.shape[1]}')
        print()

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
    
    def visual_null(self):
        plt.figure(figsize = (16, 7))

        plt.subplot(1,2,1)
        sns.heatmap(self.df.isnull(), cbar=False)
        plt.xticks(rotation = 35,
                   horizontalalignment='right',
                   fontweight='light'  )
        plt.title('Dataset missing values')
        
    def hist_str(self, column: str):
        
        dict_unique_values = dict(self.df.nunique())
        
        # set size of the plot
        plt.figure(figsize=(10, 6)) 

        # countplot shows the counts of observations in each categorical bin using bars.
        # x - name of the categorical variable
        ax = sns.countplot(x = column, data = self.df, palette=["#eed4d0", "#a2708e"])

        # set the current tick locations and labels of the x-axis.
        plt.xticks(np.arange(dict_unique_values[column]), list(self.df[column].unique()))
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
    
        plt.show()
    
    def one_numerical_dist(self, column: str):
        """[summary]

        Args:
            column (str): [description]
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
        plt.show()

    def boxplot_two_variale_str(self, column1: str, column2: str):
        '''
        column1 - категориальная переменная
        column2 - непрерывная переменная
        '''
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
        
    def numerical_target(self, column: str, target: str):
        """[summary]

        Args:
            column (str): [description]
            target (str): [description]
        """
        plt.figure(figsize=(20, 6))
        palette = sns.cubehelix_palette(5, start = 3)
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
        plt.show()

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