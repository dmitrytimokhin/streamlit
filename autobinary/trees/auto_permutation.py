import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.inspection import permutation_importance
from eli5.permutation_importance import get_score_importances

from tqdm import tqdm, tqdm_notebook
from joblib import Parallel, delayed
from collections import ChainMap


def permute(col, model: object, X: pd.DataFrame, y: pd.DataFrame, n_iter: int, metric=None, higher_is_better: bool=True, task_type: str='classification'):
    
    """
    Описание: Функция для перемешивания признака и пересчет скорра модели.
   
        model - объект модели;
        X - признаковое пространство;
        y - целевая переменная;
        n_iter - количество итераций для перемешиваний;
        metric - метрика, для перерасчета качества модели;
        higher_is_better - направленность метрики auc~True / mse~False
        task_type - тип задачи 'classification' / 'regression' / 'multiclassification'
    
    """
    
    d = {col: []}
    if task_type=='classification':       
        base_score = metric(y, model.predict_proba(X)[:, 1])
    else:
        base_score = metric(y, model.predict(X))
        
    for _ in range(n_iter):
        X_copy = X.copy()
        X_copy[col] = np.random.permutation(X_copy[col].values)
        if task_type=='classification':
            temp_prediction = model.predict_proba(X_copy)[:, 1]
        else:
            temp_prediction = model.predict(X_copy)
        score = metric(y.values, temp_prediction)
        
        if higher_is_better:
            d[col].append(base_score-score)
        else:
            d[col].append(base_score+score)
            
    return d

def kib_permute(model: object, X: pd.DataFrame, y: pd.DataFrame,
                metric=None, n_iter: int=5, n_jobs: int=-1, higher_is_better: bool=True, task_type: str='classification'):
    
    """
    Описание: Применение функции permute для формирования словаря признак - среднее значение метрики после перемешивания.
   
        model - объект модели;
        X - признаковое пространство;
        y - целевая переменная;
        metric - метрика, для перерасчета качества модели;
        n_iter - количество итераций для перемешиваний;
        n_jobs - количество ядер;
        higher_is_better - направленность метрики auc~True / mse~False;
        task_type - тип задачи 'classification' / 'regression'
    
    """    

    
    result = Parallel(n_jobs=n_jobs)(delayed(permute)(col, model, X, y, 
                                                      n_iter, metric, higher_is_better, task_type) for col in tqdm(X.columns.tolist()))
       
    
    dict_imp = dict(ChainMap(*result))
    
    for i in dict_imp.keys(): dict_imp[i] = np.mean(dict_imp[i])
    
    return dict_imp


class PermutationSelection():
    
    def __init__(self, model_type: str='xgboost', model_params: dict=None, task_type: str='classification'):
        
        """
        Описание: Класс PermutationSelection предназначен для отбора признаков. Последовательность действий выполняемых алгоритмом:
        
            1) Происходит обучение алгоритма;
            2) Происходит расчет метрики;
            3) Происходит перемешивание одного из факторов, остальные остаются неизменными;
            4) Происходит пересчет метрики с одним из перемешанных факторов;
            5) Происходит расчет разницы метрики 2) и метрики 4);
            6) Происходит повтор 5) пункта n_iter раз;
            7) Происходит усреднение пунка 6)
            8) Происходит отбор признаков либо по факторам выше значения random_feature на тесте, либо permutation importance значение на тесте > 0.

            model_type - тип обучаемого алгоритма 'xgboost' / 'catboost' / 'lightboost' / 'decisiontree' / 'randomforest';
            model_params - параметры обучаемого алгоритма;
            task_type - тип задачи 'classification' / 'regression' / 'multiclassification'
        
        """
        
        self.model_type = model_type
        self.model_params = model_params
        self.task_type = task_type
        self.random_state = self.model_params['random_state']
        
        
#    def corr_analysis(self, X_train:pd.DataFrame, cutoff:float=0.9):
#        
#        """
#        Описание: Функция corr_analysis позволяет провести корреляционный анализ факторов после трансформации (заполнение пропусков, кодирование и др.)
#            
#            X_train - признаковое пространсво;
#            cutoff - порог отсечения признаков по корреляции спирмена, default = 0.9.
#        
#        """
#        
#        corr_matrix = X_train.corr(method='spearman').abs()
#        high_corr_var = np.where(corr_matrix>cutoff)
#        high_corr_var = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
#        corr_cols_to_drop = list(set(x[1] for x in high_corr_var))
#        feat_after_corr = set(set(X_train.columns)-set(corr_cols_to_drop))
#        
#        self.notcorr_features = list(feat_after_corr)
#        
#        return list(feat_after_corr)
    
    def depth_analysis(self, X_train: pd.DataFrame, y_train: pd.DataFrame, features:list=None, max_depth:int=5):
        
        """
        Описание: Функция depth_analysis позволяет провести первоначальный анализ признаков на значимость. Просиходит обучение алгоритма с изменением глубины дерева от 1 до заданного значения. На каждом значении глубины определяется значимость факторов, далее значение по каждому фактору усредняется. Итоговым набором факторов выступают те, среднее значение которых > 0.
        
            X_train - признаковое пространство (желательно тренировочное множество, либо подмножество из тренировочного множества);
            y_train - целевая переменная;
            features - список факторов для расчета важностей с изменением глубины дерева;
            max_depth - максимальное значение глубины дерева.
        
        """
        
        max_depth_grid = list(range(1,max_depth+1))  
        fi = list()
        X_train = X_train[features].copy()
        rank_df = pd.DataFrame(X_train.columns,columns=['index']).set_index(['index'])
        
        for max_depth in tqdm_notebook(max_depth_grid):
            
            fi_feat = []
            new_params = self.model_params.copy()
            
            if self.model_type=='catboost':
                new_params['depth'] = max_depth
            else:
                new_params['max_depth'] = max_depth
            
            if self.task_type=='classification' or self.task_type=='multiclassification':
                
                if self.model_type=='xgboost':
                    model = XGBClassifier(**new_params)
                elif self.model_type=='catboost':
                    model = CatBoostClassifier(**new_params) 
                elif self.model_type=='lightboost':
                    model = LGBMClassifier(**new_params)
                elif self.model_type=='decisiontree':
                    model = DecisionTreeClassifier(**new_params)            
                elif self.model_type=='randomforest':
                    model = RandomForestClassifier(**new_params)
                    
            elif self.task_type=='regression':

                if self.model_type=='xgboost':
                    model = XGBRegressor(**new_params)
                elif self.model_type=='catboost':
                    model = CatBoostRegressor(**new_params)
                elif model_type=='lightboost':
                    model = LGBMRegressor(**new_params)
                elif model_type=='decisiontree':
                    model = DecisionTreeRegressor(**new_params)            
                elif model_type=='randomforest':
                    model = RandomForestRegressor(**new_params)   
              
            model.fit(X_train, y_train)
            
            if self.model_type=='xgboost':
                xgbimp = list(model.get_booster().get_score(importance_type='gain').values())
                fi.append(xgbimp+[i*0 for i in range(len(X_train.columns)-len(xgbimp))])
                fi_feat.append(xgbimp+[i*0 for i in range(len(X_train.columns)-len(xgbimp))])

            elif self.model_type=='catboost':
                fi.append(model.get_feature_importance())
                fi_feat.append(model.get_feature_importance())
                
            elif self.model_type=='lightboost':
                fi.append(model.booster_.feature_importance(importance_type='gain'))
                fi_feat.append(model.booster_.feature_importance(importance_type='gain'))
                
            elif self.model_type=='decisiontree' or self.model_type=='randomforest':
                fi.append(model.feature_importances_)
                fi_feat.append(model.feature_importances_)            
            
            rank = pd.DataFrame(np.array(fi_feat).T,
                              columns=['importance'],
                              index=X_train.columns).sort_values('importance', ascending=True)
            
            len_list = len(rank[rank.importance>0].index)
            rank[f'rank_depth_{max_depth}'] = [0 * i for i in range(len(rank)-len_list)]+[i/sum(range(1,len_list+1)) for i in range(1,len_list+1)]

            rank_df[f'rank_depth_{max_depth}'] = rank[f'rank_depth_{max_depth}']
        
        fi = pd.DataFrame(np.array(fi).T,
                  columns=['importance_depth_' + str(idx) for idx in range(1,len(fi)+1)],
                  index=X_train.columns)

        # вычисляем усредненные важности и добавляем столбец с ними
        fi['mean_importance'] = fi.mean(axis=1)
        rank_df['mean_rank'] = rank_df.mean(axis=1)
        
        fi['mean_rank'] = rank_df['mean_rank']
        self.deth_features_importance = fi[fi.mean_importance>0].index.tolist()
        self.deth_features_rank = fi[fi.mean_rank>0].index.tolist()
        
        depth_features = fi[fi.mean_importance>0].index.tolist()
        rank_features = fi[fi.mean_rank>0].index.tolist()
        
        print(f'Количество признаков до отбора: {len(features)}')
        print('==================================================')
        print(f'Количество признаков после mean importance относительно глубины: {len(depth_features)}')
        print(f'Количество признаков после mean rank относительно глубины: {len(rank_features)}')
        
        
        return fi.sort_values('mean_importance',ascending=False), rank_df.sort_values('mean_rank',ascending=False), depth_features, rank_features
            
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        
        """
        Описание: функция fit применяется для обучения алгоритма.
        
            X_train - признаковое пространство;
            y_train - целевая переменная;
            random_feature - создание случайного признака для отбора из существующих на тестовом множесте True / False.
        
        """
        
        
        self.X_train = X_train.copy()
        self.X_train['random_feature'] = np.random.randn(len(self.X_train))
            
        self.y_train = y_train.copy()
        
        if self.task_type=='classification' or self.task_type=='multiclassification':

            if self.model_type=='xgboost':
                self.model = XGBClassifier(**self.model_params)
            elif self.model_type=='catboost':
                self.model = CatBoostClassifier(**self.model_params) 
            elif self.model_type=='lightboost':
                self.model = LGBMClassifier(**self.model_params)
            elif self.model_type=='decisiontree':
                self.model = DecisionTreeClassifier(**self.model_params)            
            elif self.model_type=='randomforest':
                self.model = RandomForestClassifier(**self.model_params) 

        elif self.task_type=='regression':

            if self.model_type=='xgboost':
                self.model = XGBRegressor(**self.model_params)
            elif self.model_type=='catboost':
                self.model = CatBoostRegressor(**self.model_params)
            elif self.model_type=='lightboost':
                self.model = LGBMRegressor(**self.model_params)
            elif self.model_type=='decisiontree':
                self.model = DecisionTreeRegressor(**self.model_params)            
            elif self.model_type=='randomforest':
                self.model = RandomForestRegressor(**self.model_params)                
                
        self.model.fit(self.X_train, self.y_train)
        self.feature_names = self.X_train.columns.tolist()
        
    def calculate_permutation(self, X_test: pd.DataFrame, y_test: pd.DataFrame, n_iter: int=5, permute_type: str='sklearn', n_jobs: int=-1, metric=None, higher_is_better: bool=True):
        
        """
        Описание: функция calculate_permutation предназначена для расчета permutation importance.
        
            X_test - тестовое признаковое пространство;
            y_test - тестовая целевая переменная;
            n_iter - количество итераций для перемешиваний;
            permute_type - используемая библиотека для расчета permutation importance 'sklearn' / 'eli5' / 'kib';
            n_jobs - количество ядер (используется только для permutation importance от 'sklearn' и 'kib');
            metric - метрика, для перерасчета качества модели (используется только для permutation importance от 'kib');
            higher_is_better - направленность метрики auc~True / mse~False (используется только для permutation importance от 'kib').
        
        """
        
        self.permute_type = permute_type
            
        self.X_test = X_test.copy()
        
        # Создание рандомной фичи для отбора на тесте
        self.X_test['random_feature'] = np.random.randn(len(self.X_test))
            
        self.y_test = y_test
        
        # Обучение Permutation importance из разных библиотек
        if permute_type=='sklearn':
            result_tr = permutation_importance(self.model, self.X_train, self.y_train, n_repeats=n_iter, random_state=self.random_state, n_jobs=n_jobs)
            result_te = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=n_iter, random_state=self.random_state, n_jobs=n_jobs)
            
            # Создание важности и словаря факторов
            sorted_idx = result_tr.importances_mean.argsort()
            feature_names = np.array(self.feature_names)[sorted_idx]
            
            if self.task_type=='regression':
                data_tr = {'Feature':feature_names,
                           'Perm_Importance_Tr':result_tr.importances_mean[sorted_idx]*(-1)}
                data_te = {'Feature':feature_names,
                           'Perm_Importance_Te':result_te.importances_mean[sorted_idx]*(-1)}                
            else:    
                data_tr = {'Feature':feature_names,
                           'Perm_Importance_Tr':result_tr.importances_mean[sorted_idx]}
                data_te = {'Feature':feature_names,
                           'Perm_Importance_Te':result_te.importances_mean[sorted_idx]}
            
        elif permute_type=='eli5':
            _, result_tr = get_score_importances(self.model.score, self.X_train.values, self.y_train, n_iter=n_iter, random_state=self.random_state)
            _, result_te = get_score_importances(self.model.score, self.X_test.values, self.y_test, n_iter=n_iter,random_state=self.random_state)
            
            # Создание важности и словаря факторов
            if self.task_type=='regression':
                data_tr = {'Feature':self.feature_names,
                           'Perm_Importance_Tr':np.mean(result_tr, axis=0)*(-1)}
                data_te = {'Feature':self.feature_names,
                           'Perm_Importance_Te':np.mean(result_te, axis=0)*(-1)}
            else:
                data_tr = {'Feature':self.feature_names,
                           'Perm_Importance_Tr':np.mean(result_tr, axis=0)}
                data_te = {'Feature':self.feature_names,
                           'Perm_Importance_Te':np.mean(result_te, axis=0)}                
            
        elif permute_type=='kib':
            print('Расчет Permutation Importance на Train')
            result_tr = kib_permute(self.model, self.X_train, self.y_train, metric=metric, n_iter=n_iter, n_jobs=n_jobs, higher_is_better=higher_is_better, task_type=self.task_type)
            print('Расчет Permutation Importance на Test')
            result_te = kib_permute(self.model, self.X_test, self.y_test, metric=metric, n_iter=n_iter, n_jobs=n_jobs, higher_is_better=higher_is_better, task_type=self.task_type)
            
            
            
            # Создание важности и словаря факторов
            data_tr = {'Feature':result_tr.keys(),
                       'Perm_Importance_Tr':result_tr.values()}
            data_te = {'Feature':result_te.keys(),
                       'Perm_Importance_Te':result_te.values()}
        
        # Создание датасета и сортировка PI на тесте по убыванию
        self.pi_df = (pd.DataFrame(data_tr).merge(pd.DataFrame(data_te),how='left',on='Feature')).set_index('Feature').sort_values(by=['Perm_Importance_Te'], ascending=False)
        
        return self.pi_df
    
    def permutation_plot(self, top: int=None, figsize=(10,6)):
        
        """
        Описание: функция permutation_plot предназначена для отрисовки бар плота по признакам на тестовом признаковом пространстве.
        
            top - количество признаков для отрисовки бар плота. Если не указано значение, будут отрисованы все признаки, участвующие при обучении алгоритма.
        
        """
        if top is None:
            x = self.pi_df['Perm_Importance_Te']
            y = y=self.pi_df.index
        else:
            x = self.pi_df['Perm_Importance_Te'][:top]
            y = y=self.pi_df.index[:top]
            
        # Параметры для рисунка
        plt.figure(figsize=figsize)
        sns.barplot(x=x, y=y,color='dodgerblue')
        plt.title(self.permute_type + ' Feature Importance on Test')
        plt.xlabel('Permutation Importance')
        plt.ylabel('Feature Names')
    
    def select_features(self):
        
        """
        Описание: функция select_features предназначена для отбора признаков по результатам Permutation Importance.
        
            Отбор происходит по значению permutation importance > random_feature значения на тестовом множестве / значение permutation importance >= 0 на обучающем множестве / значение permutation importance >=0 на тестовом множестве.
        
        """
        
        random_score = self.pi_df.loc['random_feature'].Perm_Importance_Te

        if random_score>0:
            self.selected_features = self.pi_df[self.pi_df.Perm_Importance_Te>random_score].index.tolist()
        elif random_score<=0:
            self.selected_features = self.pi_df[self.pi_df.Perm_Importance_Te>=0].index.tolist()

        print(len(self.feature_names), 'признаков было до Permutation Importance', '\n')
        print(len(self.selected_features), 'признаков после Permutation Importance от', self.permute_type)
        
        return self.selected_features