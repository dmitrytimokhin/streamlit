import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin

class SentColumns(BaseEstimator, TransformerMixin):
    """
    Args:
        BaseEstimator ([type]): [description]
        TransformerMixin ([type]): [description]
    """
    def __init__(self, columns) -> None:
        self.columns = columns
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_x = pd.DataFrame(X)
        df_x.columns = self.columns
        return df_x
