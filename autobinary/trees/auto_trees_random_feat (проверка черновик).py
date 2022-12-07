from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import numpy as np
import pandas as pd

from tqdm import tqdm

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

class AutoTrees_my():
    def __init__(self, X_train: pd.DataFrame, y_train:pd.DataFrame, main_metric:str,
        main_estimator: object, main_fit_params: dict, main_prep_pipe: object, main_features: list, model_type:str='xgboost',
        extra_estimator: object=None, extra_fit_params:dict=None, extra_prep_pipe: object=None, extra_features: list=None, 
        solo_model: bool=False, treatment:pd.DataFrame=None, uplift:str=None, random_feature: bool=True):
        """[summary]

        Args:
            X_train (pd.DataFrame): [description]
            y_train (pd.DataFrame): [description]
            main_metric (str): [description]
            main_estimator (object): [description]
            main_fit_params (dict): [description]
            main_prep_pipe (object): [description]
            main_features (list): [description]
            extra_estimator (object, optional): [description]. Defaults to None.
            extra_fit_params (dict, optional): [description]. Defaults to None.
            extra_prep_pipe (object, optional): [description]. Defaults to None.
            extra_features (list, optional): [description]. Defaults to None.
            solo_model (bool, optional): [description]. Defaults to False.
            treatment (pd.DataFrame, optional): [description]. Defaults to None.
            uplift (str, optional): [description]. Defaults to None.
        """
        self.X_train = X_train.reset_index(drop=True).copy()
        self.y_train = y_train.reset_index(drop=True).copy()
        
        self.random_feature = random_feature
        if self.random_feature:
            self.X_train['random_feat'] = np.random.randn(len(self.X_train))
            self.main_features = main_features + ['random_feat']
        else:
            self.main_features = main_features
        
        self.main_metric = main_metric
        
        self.main_estimator = main_estimator
        self.main_fit_params = main_fit_params
        self.main_prep_pipe = main_prep_pipe

        self.model_type = model_type

        self.extra_estimator = extra_estimator
        self.extra_fit_params = extra_fit_params
        self.extra_prep_pipe = extra_prep_pipe
        self.extra_features = extra_features

        self.solo_model = solo_model
        self.treatment = treatment
        self.uplift = uplift
        
    def _get_metric(self, y_true, y_pred, metric:str):
        """[summary]

        Args:
            y_true ([type]): [description]
            y_pred ([type]): [description]
            metric (str): [description]

        Returns:
            [type]: [description]
        """
        if metric=='accuracy':
            result = accuracy_score(y_true, y_pred)
        elif metric=='roc_auc':
            result = roc_auc_score(y_true, y_pred)
        elif metric=='gini':
            result = (2*roc_auc_score(y_true, y_pred)-1)*100
        elif metric=='mae':
            result = mean_absolute_error(y_true, y_pred)
        elif metric=='mse':
            result = mean_squared_error(y_true, y_pred, squared=True)
        elif metric=='rmse':
            result = mean_squared_error(y_true, y_pred, squared=False)
        elif metric=='mape':
            result = mean_absolute_percentage_error(y_true, y_pred)
        else:
            result = np.nan
        return result

    def _calc_main_metric(self, X_train, X_test, y_train, y_test)-> dict:
        """Расчет основной метрики.
        Обычно нужна для подбора гиперпараметров с использованием Optuna.

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            y_train ([type]): [description]
            y_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if self.main_metric in ['accuracy', 'mae', 'mse', 'rmse', 'mape']:
            if self.model_type == 'xgboost':
                y_pred_train = self.main_estimator.predict(X_train, ntree_limit = self.main_estimator.get_booster().best_iteration)
                y_pred_test = self.main_estimator.predict(X_test, ntree_limit = self.main_estimator.get_booster().best_iteration)
            elif self.model_type == 'catboost':
                y_pred_train = self.main_estimator.predict(X_train)
                y_pred_test = self.main_estimator.predict(X_test)
            elif self.model_type == 'lightboost':
                y_pred_train = self.main_estimator.predict(X_train)
                y_pred_test = self.main_estimator.predict(X_test)
            elif self.model_type == 'decisiontree':
                y_pred_train = self.main_estimator.predict(X_train)
                y_pred_test = self.main_estimator.predict(X_test)
            elif self.model_type == 'randomforest':
                y_pred_train = self.main_estimator.predict(X_train)
                y_pred_test = self.main_estimator.predict(X_test)

            main_train = self._get_metric(y_true=y_train, y_pred=y_pred_train, metric=self.main_metric)
            main_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test, metric=self.main_metric)

        elif self.main_metric in ['roc_auc', 'gini', 'delta_gini']:
            if self.model_type == 'xgboost':
                y_pred_train = self.main_estimator.predict_proba(X_train, ntree_limit = self.main_estimator.get_booster().best_iteration)[:, 1]
                y_pred_test = self.main_estimator.predict_proba(X_test, ntree_limit = self.main_estimator.get_booster().best_iteration)[:, 1]
            elif self.model_type == 'catboost':
                y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
                y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
            elif self.model_type == 'lightboost':
                y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
                y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
            elif self.model_type == 'decisiontree':
                y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
                y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
            elif self.model_type == 'randomforest':
                y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
                y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
            
            if self.main_metric == 'delta_gini':
                g_train = self._get_metric(y_true=y_train, y_pred=y_pred_train, metric='gini')
                g_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test, metric='gini')

                main_train = np.nan
                main_valid = np.abs(g_train+1e-10-g_valid)/(g_valid+1e-10)

            else:
                main_train = self._get_metric(y_true=y_train, y_pred=y_pred_train, metric=self.main_metric)
                main_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test, metric=self.main_metric)

        dict_temp = {'main_train': main_train, 'main_valid': main_valid}
        
        return dict_temp
    
    def _extra_metrics_classif(self, X_train, X_test, y_train, y_test)-> dict:
        """
        Расчет дополнительных метрик для задачи класификации.

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            y_train ([type]): [description]
            y_test ([type]): [description]

        Returns:
            dict: [description]
        """

        if self.model_type == 'xgboost':
            y_pred_train = self.main_estimator.predict_proba(X_train, ntree_limit = self.main_estimator.get_booster().best_iteration)[:, 1]
            y_pred_test = self.main_estimator.predict_proba(X_test, ntree_limit = self.main_estimator.get_booster().best_iteration)[:, 1]
        elif self.model_type == 'catboost':
            y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
            y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
        elif self.model_type == 'lightboost':
            y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
            y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
        elif self.model_type == 'decisiontree':
            y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
            y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]
        elif self.model_type == 'randomforest':
            y_pred_train = self.main_estimator.predict_proba(X_train)[:, 1]
            y_pred_test = self.main_estimator.predict_proba(X_test)[:, 1]

        roc_train = self._get_metric(y_true=y_train, y_pred=y_pred_train, metric='roc_auc')
        roc_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test, metric='roc_auc')

        gini_train = self._get_metric(y_true=y_train, y_pred=y_pred_train, metric='gini')
        gini_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test, metric='gini')

        fpr_tr, tpr_tr, _ = roc_curve(y_train,  y_pred_train)
        fpr_vl, tpr_vl, _ = roc_curve(y_test,  y_pred_test)
            
        dict_temp = {'fpr_tr': fpr_tr, 'tpr_tr': tpr_tr, 'roc_train': roc_train, 'gini_train': gini_train,
                     'fpr_vl': fpr_vl, 'tpr_vl': tpr_vl, 'roc_valid': roc_valid, 'gini_valid': gini_valid}

        return dict_temp

    def _extra_metrics_regr(self, X_train, X_test, y_train, y_test)-> dict:
        """[summary]

        Args:
            X_train ([type]): [description]
            X_test ([type]): [description]
            y_train ([type]): [description]
            y_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if self.model_type == 'xgboost':
            y_pred_train = self.main_estimator.predict(X_train, ntree_limit = self.main_estimator.get_booster().best_iteration)
            y_pred_test = self.main_estimator.predict(X_test, ntree_limit = self.main_estimator.get_booster().best_iteration)
        elif self.model_type == 'catboost':
            y_pred_train = self.main_estimator.predict(X_train)
            y_pred_test = self.main_estimator.predict(X_test)
        elif self.model_type == 'lightboost':
            y_pred_train = self.main_estimator.predict(X_train)
            y_pred_test = self.main_estimator.predict(X_test)
        elif self.model_type == 'decisiontree':
            y_pred_train = self.main_estimator.predict(X_train)
            y_pred_test = self.main_estimator.predict(X_test)
        elif self.model_type == 'randomforest':
            y_pred_train = self.main_estimator.predict(X_train)
            y_pred_test = self.main_estimator.predict(X_test)

        mae_train = self._get_metric(y_true=y_train, y_pred=y_pred_train,metric='mae')
        mae_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test,metric='mae')

        mse_train = self._get_metric(y_true=y_train, y_pred=y_pred_train,metric='mse')
        mse_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test,metric='mse')

        rmse_train = self._get_metric(y_true=y_train, y_pred=y_pred_train,metric='rmse')
        rmse_valid = self._get_metric(y_true=y_test, y_pred=y_pred_test,metric='rmse')

        dict_temp = {'mae_train': mae_train, 'mse_train': mse_train, 'rmse_train': rmse_train, 
                     'mae_valid': mae_valid, 'mse_valid': mse_valid, 'rmse_valid': rmse_valid}

        return dict_temp

    def _preprocessing(self, X_tr, X_val, y_tr, y_val):
        """Применение конвейера предобработки для не аплифт модели.

        Args:
            X_tr (_type_): _description_
            X_val (_type_): _description_
            y_tr (_type_): _description_
            y_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.main_prep_pipe.fit(X_tr, y_tr)

        X_tr = self.main_prep_pipe.transform(X_tr).reset_index(drop=True)
        X_val = self.main_prep_pipe.transform(X_val).reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        return X_tr, X_val, y_tr, y_val

    def _solo_preprocessing(self, X_tr, X_val, y_tr, y_val, trt_tr=None, trt_val=None):
        """Применение конвейера для соло-аплифт моделей.

        Args:
            X_tr (_type_): _description_
            X_val (_type_): _description_
            y_tr (_type_): _description_
            y_val (_type_): _description_
            trt_tr (_type_, optional): _description_. Defaults to None.
            trt_val (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        X_tr, X_val, y_tr, y_val = self._preprocessing(
                X_tr=X_tr, X_val=X_val, y_tr=y_tr, y_val=y_val)

        if trt_tr is not None:
            trt_tr = trt_tr.reset_index(drop=True)
            trt_val = trt_val.reset_index(drop=True)
            X_tr = X_tr.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            y_tr = y_tr.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)

        if self.uplift == 'solo_dummy':
            X_tr = X_tr.assign(treatment=trt_tr)
            X_val = X_val.assign(treatment=trt_val)
        
        elif self.uplift == 'solo_interaction':
            X_tr = pd.concat([
                X_tr,
                X_tr.apply(lambda x: x * trt_tr)
                .rename(columns=lambda x: str(x) + '_treatment_interaction')], axis=1).assign(treatment=trt_tr)
            X_val = pd.concat([
                X_val,
                X_val.apply(lambda x: x * trt_val)
                .rename(columns=lambda x: str(x) + '_treatment_interaction')], axis=1).assign(treatment=trt_val)

            ##self._kk = X_tr

        elif self.uplift == 'solo_classtrans':
            y_tr = (np.array(y_tr) == np.array(trt_tr)).astype(int)
            y_val = (np.array(y_val) == np.array(trt_val)).astype(int)
        
        return X_tr, X_val, y_tr, y_val

    def _two_preprocessing(self, X_tr, X_val, y_tr, y_val, trt_tr=None, trt_val=None):
        """Применение конвейера для не two-models-аплифт моделей.

        Args:
            X_tr (_type_): _description_
            X_val (_type_): _description_
            y_tr (_type_): _description_
            y_val (_type_): _description_
            trt_tr (_type_, optional): _description_. Defaults to None.
            trt_val (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if trt_tr is not None:
            trt_tr = trt_tr.reset_index(drop=True)
            trt_val = trt_val.reset_index(drop=True)
            X_tr = X_tr.reset_index(drop=True)
            X_val = X_val.reset_index(drop=True)
            y_tr = y_tr.reset_index(drop=True)
            y_val = y_val.reset_index(drop=True)

        if self.uplift == 'two_vanilla_ctrl':
            X_tr, y_tr = X_tr[trt_tr == 0], y_tr[trt_tr == 0]
            X_val, y_val = X_val[trt_val == 0], y_val[trt_val == 0]

            X_tr, X_val, y_tr, y_val = self._preprocessing(
                X_tr=X_tr, X_val=X_val, y_tr=y_tr, y_val=y_val)
        
        elif self.uplift == 'two_vanilla_trt':
            X_tr, y_tr = X_tr[trt_tr == 1], y_tr[trt_tr == 1]
            X_val, y_val = X_val[trt_val == 1], y_val[trt_val == 1]

            X_tr, X_val, y_tr, y_val = self._preprocessing(
                X_tr=X_tr, X_val=X_val, y_tr=y_tr, y_val=y_val)
        
        elif self.uplift == 'two_ddr_control':
            X_ctrl_tr, y_ctrl_tr = X_tr[trt_tr == 0], y_tr[trt_tr == 0]
            X_ctrl_val, y_ctrl_val = X_val[trt_val == 0], y_val[trt_val == 0]

            X_ctrl_tr, X_ctrl_val, y_ctrl_tr, y_ctrl_val = self._preprocessing(
                X_tr=X_ctrl_tr, X_val=X_ctrl_val, y_tr=y_ctrl_tr, y_val=y_ctrl_val)

            X_trt_tr, y_trt_tr = X_tr[trt_tr == 1], y_tr[trt_tr == 1]
            X_trt_val, y_trt_val = X_val[trt_val == 1], y_val[trt_val == 1]

            X_trt_tr, X_trt_val, y_trt_tr, y_trt_val = self._preprocessing(
                X_tr=X_trt_tr, X_val=X_trt_val, y_tr=y_trt_tr, y_val=y_trt_val)

            self.extra_fit_params.update({
                'X':X_ctrl_tr, 
                'y':y_ctrl_tr, 
                'eval_set':[(X_ctrl_tr, y_ctrl_tr), (X_ctrl_val, y_ctrl_val)]
            })
            
            print(f'*************** Обучение DDR-control ***************')

            self.extra_estimator.fit(
                X_ctrl_tr, y_ctrl_tr, **self.extra_fit_params)

            ddr_control_tr = self.extra_estimator.predict_proba(X_trt_tr)[:, 1]
            ddr_control_val = self.extra_estimator.predict_proba(X_trt_val)[:, 1]

            print(f'*************** Обучение DDR-control окончено ***************')

            X_tr = X_trt_tr.assign(ddr_control=ddr_control_tr)
            X_val = X_trt_val.assign(ddr_control=ddr_control_val)
            y_tr = y_trt_tr
            y_val = y_trt_val

        elif self.uplift == 'two_ddr_treatment':
            X_ctrl_tr, y_ctrl_tr = X_tr[trt_tr == 0], y_tr[trt_tr == 0]
            X_ctrl_val, y_ctrl_val = X_val[trt_val == 0], y_val[trt_val == 0]

            X_ctrl_tr, X_ctrl_val, y_ctrl_tr, y_ctrl_val = self._preprocessing(
                X_tr=X_ctrl_tr, X_val=X_ctrl_val, y_tr=y_ctrl_tr, y_val=y_ctrl_val)

            X_trt_tr, y_trt_tr = X_tr[trt_tr == 1], y_tr[trt_tr == 1]
            X_trt_val, y_trt_val = X_val[trt_val == 1], y_val[trt_val == 1]

            X_trt_tr, X_trt_val, y_trt_tr, y_trt_val = self._preprocessing(
                X_tr=X_trt_tr, X_val=X_trt_val, y_tr=y_trt_tr, y_val=y_trt_val)

            self.extra_fit_params.update({
                'X':X_trt_tr, 
                'y':y_trt_tr, 
                'eval_set':[(X_trt_tr, y_trt_tr), (X_trt_val, y_trt_val)]
            })
            
            print(f'*************** Обучение DDR-treatment ***************')

            self.extra_estimator.fit(
                X_trt_tr, y_trt_tr, **self.extra_fit_params)

            ddr_treatment_tr = self.extra_estimator.predict_proba(X_ctrl_tr)[:, 1]
            ddr_treatment_val = self.extra_estimator.predict_proba(X_ctrl_val)[:, 1]

            print(f'*************** Обучение DDR-treatment окончено ***************')

            X_tr = X_ctrl_tr.assign(ddr_control=ddr_treatment_tr)
            X_val = X_ctrl_val.assign(ddr_control=ddr_treatment_val)
            y_tr = y_ctrl_tr
            y_val = y_ctrl_val
    
        return X_tr, X_val, y_tr, y_val

    def _model_fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_val=None, y_val=None,
                   trt_train=None, trt_val=None):
        """Обучение одной модели.

        Args:
            X_train (pd.DataFrame): [description]
            y_train ([type]): [description]
            X_val (pd.DataFrame): [description]
            y_val (pd.DataFrame): [description]

        Returns:
            [type]: [description]
        """

        # если не аплифт модель, то просто применяем конвейер
        if self.uplift is None:
            X_train_new, X_val_new, y_train, y_val = self._preprocessing(
                X_tr=X_train, X_val=X_val, y_tr=y_train, y_val=y_val)

        elif self.uplift in ['solo_dummy', 'solo_interaction', 'solo_classtrans']:
            X_train_new, X_val_new, y_train, y_val = self._solo_preprocessing(
                X_tr=X_train, X_val=X_val, y_tr=y_train, y_val=y_val,
                trt_tr=trt_train, trt_val=trt_val)

        elif self.uplift in ['two_vanilla_ctrl', 'two_vanilla_trt', 'two_ddr_control', 'two_ddr_treatment']:
            X_train_new, X_val_new, y_train, y_val = self._two_preprocessing(
                X_tr=X_train, X_val=X_val, y_tr=y_train, y_val=y_val,
                trt_tr=trt_train, trt_val=trt_val)

        # обновляем параметры обучения для новых выборок
        if self.model_type in ['xgboost', 'catboost', 'lightboost']:
            if self.solo_model is False:
                # Обновляем параметры обучения
                self.main_fit_params.update({
                    'X':X_train_new, 
                    'y':y_train, 
                    'eval_set':[(X_train_new, y_train), (X_val_new, y_val)]
                    })

        elif self.model_type in ['decisiontree', 'randomforest']:
            # Обновляем параметры обучения
            self.main_fit_params.update({
                'X':X_train_new, 
                'y':y_train
                })

        # Обучаем модель на трейне
        self.main_estimator.fit(**self.main_fit_params)

        if self.model_type == 'xgboost':
            print('BEST ITERATION: ', self.main_estimator.get_booster().best_iteration)

            best_iteration = self.main_estimator.get_booster().best_iteration
            feature_imp = self.main_estimator.get_booster().get_score(importance_type='gain')
            evals = self.main_estimator.evals_result()
        
        elif self.model_type == 'catboost':
            print('BEST ITERATION: ', self.main_estimator.get_best_iteration())

            best_iteration = self.main_estimator.get_best_iteration()
            feature_imp = self.main_estimator.get_feature_importance(prettified=True, fstr_type='PredictionValuesChange')
            evals = self.main_estimator.get_evals_result()
        
        elif self.model_type == 'lightboost':
            print('BEST ITERATION: ', self.main_estimator.best_iteration_)
            
            best_iteration = self.main_estimator.best_iteration_
            feature_imp = self.main_estimator.booster_.feature_importance(importance_type='gain')
            evals = self.main_estimator.evals_result_            
            
        elif self.model_type in ['decisiontree', 'randomforest']:
            best_iteration = np.nan
            feature_imp = self.main_estimator.feature_importances_
            evals = np.nan
        
        # формируем предсказания для тестовой выборки
        if self.main_metric in ['roc_auc', 'gini', 'delta_gini']:
            if self.model_type == 'xgboost':
                x_test_predict = self.main_estimator.predict_proba(X_val_new, ntree_limit = self.main_estimator.get_booster().best_iteration)[:,1]
            elif self.model_type == 'catboost':
                x_test_predict = self.main_estimator.predict_proba(X_val_new)[:,1]
            elif self.model_type == 'lightboost':
                x_test_predict = self.main_estimator.predict_proba(X_val_new)[:,1]
            elif self.model_type in ['decisiontree', 'randomforest']:
                x_test_predict = self.main_estimator.predict_proba(X_val_new)[:,1]

        elif self.main_metric in ['accuracy', 'mae', 'mse', 'rmse', 'mape']:
            if self.model_type == 'xgboost':
                x_test_predict = self.main_estimator.predict(X_val_new, ntree_limit = self.main_estimator.get_booster().best_iteration)
            elif self.model_type == 'catboost':
                x_test_predict = self.main_estimator.predict(X_val_new)
            elif self.model_type == 'lightboost':
                x_test_predict = self.main_estimator.predict(X_val_new)
            elif self.model_type in ['decisiontree', 'randomforest']:
                x_test_predict = self.main_estimator.predict(X_val_new)


        main_metric = self._calc_main_metric(X_train = X_train_new, X_test=X_val_new, y_train=y_train, y_test=y_val)

        if self.main_metric in ['roc_auc', 'accuracy', 'gini', 'delta_gini']:
            metr = self._extra_metrics_classif(X_train = X_train_new, X_test=X_val_new, y_train=y_train, y_test=y_val)
        elif self.main_metric in ['mae', 'mse', 'rmse', 'mape']:
            metr = self._extra_metrics_regr(X_train = X_train_new, X_test=X_val_new, y_train=y_train, y_test=y_val)

        if self.solo_model is True:
            self.feature_imp = feature_imp
            self.metr = metr
            self.evals = evals
            self.main_metric = main_metric

            return x_test_predict, best_iteration
        else:
            return best_iteration, feature_imp, metr, evals, main_metric

    def model_fit_cv(self, strat, groups=None):
        """Функция кросс-валидации.

        Args:
            strat ([type]): [description]
            groups ([type], optional): [description]. Defaults to None.
        """

        i=0
        
        self._test_group = []

        self._best_iters = []
        self._fi = []
        self._extra_scores = {}
        self._boost_logs = {}
        self._main_scores = {}

        X_train_global = self.X_train[self.main_features].reset_index(drop=True).copy()
        y_train_global = self.y_train.reset_index(drop=True).copy()

        if self.treatment is not None:
            treatment = self.treatment.reset_index(drop=True).copy()

        if groups is not None:
            groups = self.groups.reset_index(drop=True).copy()

        for (train_index, test_index) in tqdm(strat.split(X_train_global, y_train_global, groups=groups)):
            i+=1

            print(f'==================== Обучение {i} фолда! ====================')
            if groups is not None:
                groups_train, groups_test = groups.loc[train_index], groups.loc[test_index]
                self._test_group.append(len(set(groups_train).intersection(set(groups_test))))

            if self.treatment is not None:
                trt_train, trt_test = treatment.loc[train_index], treatment.loc[test_index]
            else:
                trt_train=None
                trt_test=None

            X_train, X_test = X_train_global.iloc[train_index],X_train_global.iloc[test_index]
            y_train, y_test = y_train_global.iloc[train_index],y_train_global.iloc[test_index]

            # модель с контролем переобучения на последнем фолде
            best_iter, imp, metr, evals, main_metric = self._model_fit(
                X_train=X_train, 
                y_train=y_train,
                X_val=X_test, 
                y_val=y_test,
                trt_train=trt_train,
                trt_val=trt_test)

            self._best_iters.append(best_iter)
            self._fi.append(imp)
            self._extra_scores[f'scores_{i}'] = metr
            self._boost_logs[f'evals_{i}'] = evals
            self._main_scores[f'scores_{i}'] = main_metric

            print(f'{self.main_metric} '+'на обучающей выборке: {:.3f}'.format(main_metric['main_train']))
            print(f'{self.main_metric} '+'на проверочной выборке: {:.3f}'.format(main_metric['main_valid']))
            
            print()
            print(f'====================================================================================')

    def get_mean_cv_scores(self)-> float:
        """[summary]

        Returns:
            float: [description]
        """
        val_metrics = []
        for i in range(1, len(self._main_scores)+1):
            val_metrics.append(self._main_scores[f'scores_{i}']['main_valid'])
    
        return np.mean(val_metrics)

    def get_extra_scores(self)->pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """

        df = pd.DataFrame()
        if self.main_metric in ['roc_auc', 'accuracy', 'gini', 'delta_gini']:
            df['names'] = ['roc_train', 'roc_valid', 'gini_train', 'gini_valid']

            for i in range(1, len(self._extra_scores)+1):
                val_metrics = []
                val_metrics.append(self._extra_scores[f'scores_{i}']['roc_train'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['roc_valid'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['gini_train'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['gini_valid'])
                df[f'fold_{i}'] = val_metrics   
        elif self.main_metric in ['mae', 'mse', 'rmse', 'mape']:
            df['names'] = ['mae_train', 'mae_valid', 'mse_train', 'mse_valid', 'rmse_train', 'rmse_valid']

            for i in range(1, len(self._extra_scores)+1):
                val_metrics = []
                val_metrics.append(self._extra_scores[f'scores_{i}']['mae_train'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['mae_valid'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['mse_train'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['mse_valid'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['rmse_train'])
                val_metrics.append(self._extra_scores[f'scores_{i}']['rmse_valid'])
                df[f'fold_{i}'] = val_metrics   

        return df


    def get_fi(self)->pd.DataFrame:
        """Получить список важностей факторов из модели.

        Returns:
            pd.DataFrame: [description]
        """

        if self.solo_model is True:
            df_fi = pd.DataFrame()
            df_fi['index'] = list(self.feature_imp.keys())
            df_fi['importance'] = list(self.feature_imp.values())
            df_fi = df_fi.fillna(0)
            df_fi = df_fi.sort_values('importance', ascending=False).reset_index(drop=True)
            
            return df_fi
        else:
            if self.model_type == 'xgboost':
                df_fi = pd.DataFrame(self._fi).T
                df_fi = df_fi.fillna(0)
                df_fi.columns = ['importance '+ str(idx) for idx in range(len(self._fi))]

                # получаем усредненные важности признаков и выводим в порядке убывания
                df_fi['mean_importance'] = df_fi.mean(axis=1)
                df_fi = df_fi.sort_values('mean_importance', ascending=False)
                df_fi = df_fi.reset_index()
            
            elif self.model_type == 'catboost':
                df_fi = self._fi[0]
                df_fi.columns = ['Feature Id', 'importance_0']

                for i in range(1, len(self._fi)):
                    df1 = self._fi[i]
                    df1.columns = ['Feature Id', f'importance_{i}']
                    df_fi = df_fi.merge(
                        df1,
                        left_on = 'Feature Id',
                        right_on = 'Feature Id',
                        how='left')
                    
                filter_col = [col for col in df_fi if col.startswith('importance_')]
                df_fi['mean_importance'] = df_fi[filter_col].mean(axis=1)
                df_fi = df_fi.sort_values('mean_importance', ascending=False).reset_index(drop=True)
                df_fi = df_fi.rename(columns ={'Feature Id':'index'})

            elif self.model_type == 'lightboost':
                df_fi = pd.DataFrame(self._fi).T
                df_fi = df_fi.fillna(0)
                df_fi.index = self.main_estimator.feature_name_
                df_fi.columns = ['importance '+ str(idx) for idx in range(len(self._fi))]

                # получаем усредненные важности признаков и выводим в порядке убывания
                df_fi['mean_importance'] = df_fi.mean(axis=1)
                df_fi = df_fi.sort_values('mean_importance', ascending=False)
                df_fi = df_fi.reset_index()
            elif self.model_type in ['decisiontree', 'randomforest']:
                df_fi = pd.DataFrame(self._fi)
                df_fi = df_fi.fillna(0)
                df_fi.columns = self.main_features
                df_fi = df_fi.T
                df_fi.columns = ['importance '+ str(idx) for idx in range(len(self._fi))]

                df_fi['mean_importance'] = df_fi.mean(axis=1)
                df_fi = df_fi.sort_values('mean_importance', ascending=False)
                df_fi = df_fi.reset_index()

            return df_fi

    def _one_curve_plot(self, fig, data, metric, row, col, best_iter):
        
        if row==1 & col ==1:
            showlegend=True
        else:
            showlegend=False
        
        if self.model_type == 'lightboost':
            x_axis = list(range(0, len(data['training'][metric])))
            y_axis = data['training'][metric]
        else:
            x_axis = list(range(0, len(data['validation_0'][metric])))
            y_axis = data['validation_0'][metric]
        fig.add_trace(
        go.Scatter(
            x = x_axis,
            y = y_axis,
            mode = "lines",
            name = "Обучающая выборка",
            marker = dict(color = 'rgba(0, 197, 255, 1)'),
            text= 'Обучающая выборка',
            legendgroup = '1',
            showlegend=showlegend
            ),
        row=row, col=col
        )
        
        if self.model_type == 'lightboost':
            y_axis = [min(data['training'][metric]), max(data['valid_1'][metric])]
        else:
            y_axis = [min(data['validation_0'][metric]), max(data['validation_1'][metric])]
        
        fig.add_trace(
        go.Scatter(
            x = [best_iter, best_iter],
            y = y_axis,
            mode = "lines",
            marker = dict(color = 'rgba(0, 0, 0, 1)'),
            showlegend=False
            ),
        row=row, col=col
        )

        if self.model_type == 'lightboost':
            x_axis = list(range(0, len(data['valid_1'][metric])))
            y_axis = data['valid_1'][metric]
        else:
            x_axis = list(range(0, len(data['validation_1'][metric])))
            y_axis = data['validation_1'][metric]
        
        fig.add_trace(
        go.Scatter(
            x = x_axis,
            y = y_axis,
            mode = "lines",
            name = "Проверочная выборка",
            marker = dict(color = 'rgba(255, 154, 0, 1)'),
            text= 'Проверочная выборка',
            legendgroup = '1',
            showlegend=showlegend
            ),
        row=row, col=col
        )

        if self.model_type == 'lightboost':
            y_axis = [min(data['valid_1'][metric]), max(data['training'][metric])]
        else:
            y_axis = [min(data['validation_1'][metric]), max(data['validation_0'][metric])]       
        
        fig.add_trace(
        go.Scatter(
            x = [best_iter, best_iter],
            y = y_axis,
            mode = "lines",
            marker = dict(color = 'rgba(0, 0, 0, 1)'),
            showlegend=False
            ),
        row=row, col=col
        )

    def get_curve_plots(self):
        
        if self.model_type == 'lightboost':
            metrics = list(self._boost_logs['evals_1']['training'].keys())
        else:
            metrics = list(self._boost_logs['evals_1']['validation_0'].keys())
        epochs = len(self._boost_logs)
        
        fig = make_subplots(
            rows=epochs, 
            cols=len(metrics), 
            subplot_titles=(metrics))

        for i in range(0, epochs):
            for j in range(0, len(metrics)):
                self._one_curve_plot(fig=fig, data=self._boost_logs[f'evals_{i+1}'], metric=metrics[j], row=i+1, col=j+1, best_iter=self._best_iters[i])
                
        for i in range(0, epochs):
            for j in range(0, len(metrics)):
                fig.update_xaxes(title_text="Iterations", row=i+1, col=j+1)
                fig.update_yaxes(title_text=metrics[j] + f' - FOLD {i+1}', row=i+1, col=j+1)
                
        fig.update_layout(height=500*epochs, width=2000)
        fig.show()


    def get_rocauc_plots(self):
        """
        """
        fig, ax = plt.subplots()
        for i in range(1, len(self._extra_scores)+1):
            plt.plot(self._extra_scores[f'scores_{i}']['fpr_tr'], 
                     self._extra_scores[f'scores_{i}']['tpr_tr'], 
                     label="fold {}, AUC={:.3f}".format(i, self._extra_scores[f'scores_{i}']['roc_train']))
    
        plt.plot([0,1], [0,1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve Analysis TRAIN', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.grid()
        plt.tight_layout()
        #plt.show()
        
        fig, ax = plt.subplots()
        for i in range(1, len(self._extra_scores)+1):
            plt.plot(self._extra_scores[f'scores_{i}']['fpr_vl'], 
                     self._extra_scores[f'scores_{i}']['tpr_vl'], 
                     label="fold {}, AUC={:.3f}".format(i, self._extra_scores[f'scores_{i}']['roc_valid']))
    
        plt.plot([0,1], [0,1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve Analysis VALIDATION', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.grid()
        plt.tight_layout()
        #plt.show()
        