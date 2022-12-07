from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from .sent_columns import SentColumns
from autobinary.libraries.category_encoders.cat_boost import CatBoostEncoder

def base_pipe(num_columns:list=None, cat_columns:list=None, kind:str='all', fill_value:float=-1e24):

    if kind == 'all' or kind == 'num':
        # создаем конвейер для количественных переменных
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value = fill_value))
        ])

    if kind == 'all' or kind == 'cat':
        # создаем конвейер для категориальных переменных
        cat_pipe = Pipeline([
            ('catenc', CatBoostEncoder(cols=cat_columns))
        ])

    if kind == 'all':
        transformers = [('num', num_pipe, num_columns),
                        ('cat', cat_pipe, cat_columns)]

        # передаем список трансформеров в ColumnTransformer
        transformer = ColumnTransformer(transformers=transformers)

        # задаем итоговый конвейер
        prep_pipe = Pipeline([
            ('transform', transformer),
            ('sent_columns', SentColumns(columns=num_columns+cat_columns))
        ])

    elif kind == 'num':
        transformers = [('num', num_pipe, num_columns)]

        # передаем список трансформеров в ColumnTransformer
        transformer = ColumnTransformer(transformers=transformers)

        # задаем итоговый конвейер
        prep_pipe = Pipeline([
            ('transform', transformer),
            ('sent_columns', SentColumns(columns=num_columns))
        ])

    elif kind == 'cat':
        transformers = [('cat', cat_pipe, cat_columns)]

        # передаем список трансформеров в ColumnTransformer
        transformer = ColumnTransformer(transformers=transformers)

        # задаем итоговый конвейер
        prep_pipe = Pipeline([
            ('transform', transformer),
            ('sent_columns', SentColumns(columns=cat_columns))
        ])

    return prep_pipe
