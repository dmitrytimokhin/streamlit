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
    
    def base_overview(self):

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
    
        