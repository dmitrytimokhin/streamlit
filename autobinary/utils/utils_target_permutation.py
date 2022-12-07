import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# вспомогательный функции для использования и тестирования метода target permutation

# функция для инита естиматора (для удобного запуска в цикле по разным эстиматорам)
def select_estimator(task_type, model_type, main_fit_params):
    
    if task_type=='classification':

        if model_type=='xgboost':
            model = XGBClassifier(**main_fit_params)
        elif model_type=='catboost':
            model = CatBoostClassifier(**main_fit_params) 
        elif model_type=='lightboost':
            model = LGBMClassifier(**main_fit_params)
        elif model_type=='decisiontree':
            model = DecisionTreeClassifier(**main_fit_params)            
        elif model_type=='randomforest':
            model = RandomForestClassifier(**main_fit_params) 

    if task_type == 'regression':

        if model_type=='xgboost':
            model = XGBRegressor(**main_fit_params)
        elif model_type=='catboost':
            model = CatBoostRegressor(**main_fit_params)
        elif model_type=='lightboost':
            model = LGBMRegressor(**main_fit_params)
        elif model_type=='decisiontree':
            model = DecisionTreeRegressor(**main_fit_params)            
        elif model_type=='randomforest':
            model = RandomForestRegressor(**main_fit_params)                

    return model


# функция для генерации весов (геометрической регрессией)
def geom_reg_list(n: int, b1: float, q: float = 9/10):
    return [b1*q**(i-1) for i in range(1,n+1)]


# функция для взвешенного аггрегированного отбора признаков с попомщью результатов нескольких моделей
def aggregated_feature_selection(feat_dist: dict, weights_list_feat: list = None, freq: int = None, mean_score_select: bool = True, num_feat: int = None):
    
    if weights_list_feat == None:
        print('Веса для первый фичи для каждой модели выбраны одинаковыми => все модели имеют одинаковый вес.')
        weights_list_feat = [100/len(feat_dist.keys()) for i in feat_dist.keys()]
    all_variants = set([j for sub in feat_dist.values() for j in sub])
    weights_dict_feat = dict((key,0) for key in all_variants)
    frequency_dict_feat = dict((key,0) for key in all_variants)

    for idx, fi_list in enumerate(feat_dist.values()):
        weights_list_model = geom_reg_list(n=len(fi_list), b1=weights_list_feat[idx])
        
        for count, elem in enumerate(fi_list):
            weights_dict_feat[elem] += weights_list_model[count]
            frequency_dict_feat[elem] += 1
            
        pd_weights_feat = pd.DataFrame()
        pd_weights_feat['features'] = weights_dict_feat.keys()
        pd_weights_feat['score'] = weights_dict_feat.values()
        pd_weights_feat = pd_weights_feat.sort_values(by=['score'], ascending=False)
        
    if mean_score_select:
        print('Отобраны фичи, которые имеют скор больше среднего.')
        pd_weights_feat = pd_weights_feat[pd_weights_feat['score'] >= pd_weights_feat['score'].mean()]
        
    if freq != None:
        print(f'Отобраны фичи, которые имеются в {freq} моделях.')
        feat_list = []
        for feat, frequency in frequency_dict_feat.items():
            if frequency >= freq:
                feat_list.append(feat)
        pd_weights_feat = pd_weights_feat[pd_weights_feat['features'].isin(feat_list)]
        
    if num_feat != None:
        print(f'Отобраны {num_feat} фичей с лучшим скором')
        pd_weights_feat = pd_weights_feat.iloc[[i for i in range(num_feat)]]
    else:
        print(f'Отобрано {len(pd_weights_feat)} фичей')
        
    return pd_weights_feat    


# функция для вывода пересечений множеств отобраных разными эстиматорами фичей 
def intersection_set(feat_dist: dict, freq: int = 3) -> set:
    all_variants = set([j for sub in feat_dist.values() for j in sub])
    drop_set = []
    idx = 0
    for col in all_variants:
        idx = 0
        for fi_dict in feat_dist.values():
            if col in fi_dict:
                idx+=1
        if idx < freq:
            drop_set.append(col)
    return set([j for sub in feat_dist.values() for j in sub]).difference(drop_set)