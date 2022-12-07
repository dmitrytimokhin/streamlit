from .auto_trees import *

def _get_score_selection(X, estimator, model_type, metric):

    if metric in ['roc_auc', 'gini', 'delta_gini']:
        if model_type == 'xgboost':
            y_predict = estimator.predict_proba(X, ntree_limit = self.main_estimator.get_booster().best_iteration)
        else:
            y_predict = estimator.predict_proba(X)[:,1]

    elif metric in ['accuracy', 'mae', 'mse', 'rmse', 'mape', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_weighted']:
        if model_type == 'xgboost':
            y_predict = estimator.predict(X, ntree_limit = self.main_estimator.get_booster().best_iteration)
        else:
            y_predict = estimator.predict(X)

    elif metric in ['roc_auc_ovr', 'roc_auc_ovo']:
        if model_type == 'xgboost':
            y_predict = estimator.predict_proba(X, ntree_limit = self.main_estimator.get_booster().best_iteration)
        else:
            y_predict = estimator.predict_proba(X)
            
    return y_predict

class AutoSelection(AutoTrees):
    def __init__(self, X_train:pd.DataFrame, y_train:pd.DataFrame, 
                 main_metric:str,
                 main_estimator:object, main_fit_params:dict,
                 model_type:str='xgboost',
                 extra_estimator:object=None, extra_fit_params:dict=None, 
                 extra_prep_pipe:object=None, extra_features:list=None, 
                 solo_model:bool=False, treatment:pd.DataFrame=None, uplift:str=None,
                 base_pipe:object=None, num_columns:list=None, cat_columns:list=None
                 ):
        
        self.X_train = X_train.reset_index(drop=True).copy()
        self.y_train = y_train.reset_index(drop=True).copy()

        self.num_columns = num_columns
        self.cat_columns = cat_columns        
        self.main_features = num_columns+cat_columns
        self.main_metric = main_metric
        
        self.main_estimator = main_estimator
        self.main_fit_params = main_fit_params
        self.base_pipe = base_pipe
        self.main_prep_pipe = self.base_pipe(num_columns=self.num_columns, cat_columns=self.cat_columns)

        self.model_type = model_type

        self.extra_estimator = extra_estimator
        self.extra_fit_params = extra_fit_params
        self.extra_prep_pipe = extra_prep_pipe
        self.extra_features = extra_features

        self.solo_model = solo_model
        self.treatment = treatment
        self.uplift = uplift
        self.forward = False
        self.backward = False
        self.dbackward = False
        
    def forward_selection(self, strat, groups=None):
        
        self.main_features = self.num_columns+self.cat_columns
        self.main_prep_pipe = self.base_pipe(num_columns=self.num_columns, cat_columns=self.cat_columns)
        self.model_fit_cv(strat=strat,groups=groups)
        self.mean_metric_folds = list()
        for i in range(1,len(self._main_scores.keys())+1):
            self.mean_metric_folds.append(self._main_scores[f'scores_{i}']['main_valid'])
        self.mean_metric_folds = np.array(self.mean_metric_folds)
        fi = self.get_fi()
        features_imp = fi[fi['mean_importance']>0]['index'].to_list()
        
        fselection_res = dict()
        fselection_res['features_stack'] = None
        
        print('Средняя метрика на фолдах со всеми факторами: ',np.round(np.mean(self.mean_metric_folds),6))
        print('==========================')
        print('Конец обуения кросс-валидации!')
        print()
        
        print('Начало жадного отбора факторов Forward Selection!')
        print('==========================')

        features_stack = list()
        features_scores = list()
        for i, feature in enumerate(features_imp):
            features_stack.append(feature)
            _fold_scores = list()
            print(f'Добавление признака {feature}')
            
            num_stack = list(filter(lambda x: x in self.num_columns, features_stack))
            cat_stack = list(filter(lambda x: x in self.cat_columns, features_stack))

            self.main_features = num_stack+cat_stack
            
            if len(num_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='cat')
            elif len(cat_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='num')
            else:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='all')
            
            self.model_fit_cv(strat=strat, groups=groups)
            
            _fold_scores = list()
            for i in range(1,len(self._main_scores.keys())+1):
                _fold_scores.append(self._main_scores[f'scores_{i}']['main_valid'])
            _fold_scores = np.array(_fold_scores)
            mean_fold_scores = np.mean(_fold_scores)
            
            features_scores.append(mean_fold_scores)
            print(f'Количество признаков: {len(features_stack)} => метрика: {mean_fold_scores}')
            
            
        
            # Условие стопа (если alt лучше на более, чем половине фолдов)
            if self.main_metric in ['accuracy','roc_auc', 'gini', 'delta_gini', 'roc_auc_ovr', 'roc_auc_ovo', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_weighted']:
                if (np.sum(_fold_scores >= self.mean_metric_folds) > strat.n_splits // 2):

                    fselection_res['features_stack'] = features_stack
                    fselection_res['num_features_stack'] = num_stack
                    fselection_res['cat_features_stack'] = cat_stack
                    fselection_res['features_scores'] = features_scores
                    fselection_res['metric_alt'] = np.round(mean_fold_scores, 6)

                    break    
                else:
                    fselection_res['features_stack'] = features_stack
                    fselection_res['num_features_stack'] = num_stack
                    fselection_res['cat_features_stack'] = cat_stack
                    fselection_res['features_scores'] = features_scores
                    fselection_res['metric_alt'] = np.round(mean_fold_scores, 6)
                    
            elif self.main_metric in ['mae', 'mse', 'rmse', 'mape']:
                if (np.sum(_fold_scores <= self.mean_metric_folds) > strat.n_splits // 2):

                    fselection_res['features_stack'] = features_stack
                    fselection_res['num_features_stack'] = num_stack
                    fselection_res['cat_features_stack'] = cat_stack
                    fselection_res['features_scores'] = features_scores
                    fselection_res['metric_alt'] = np.round(mean_fold_scores, 6)

                    break    
                else:
                    fselection_res['features_stack'] = features_stack
                    fselection_res['num_features_stack'] = num_stack
                    fselection_res['cat_features_stack'] = cat_stack
                    fselection_res['features_scores'] = features_scores
                    fselection_res['metric_alt'] = np.round(mean_fold_scores, 6)
            print('==========================')
        print()
        print(f'Количество отобранных признаков: {len(features_stack)}')
        print(f'Метрика до отбора: {np.round(np.mean(self.mean_metric_folds),4)} => после отбора: {np.round(fselection_res["metric_alt"],4)}')
        print('==========================')
        print(f'Конец жадного отбора факторов Forward Selection!')

        self.fselection_res = fselection_res
        self.forward = True
        
        return fselection_res

    def plot_forward(self, save:bool=False, figsize=(12, 8)):
        fig = plt.figure(figsize=figsize)
        plt.ylabel(self.main_metric)
        plt.title('Прямой последовательный отбор')
        plt.plot(self.fselection_res['features_stack'], self.fselection_res['features_scores'], linewidth=2, markersize=10, marker='s', linestyle='-', label = 'Скор последовательного отбора')
        plt.plot([np.mean(self.mean_metric_folds) for c in self.fselection_res['features_scores']], linewidth=4, linestyle=':', label = f'Среднее значение метрики на кросс-валидации = {np.round(np.mean(self.mean_metric_folds),3)}')
        plt.legend()
        plt.xticks(rotation=90)
        plt.grid()
        if save:
            plt.savefig('forward_{}.png'.format(self.main_metric), dpi=300)
        plt.show()    
        
    def backward_selection(self, strat, groups=None, first_degradation:bool=True):
        
        self.main_features = self.num_columns+self.cat_columns
        self.main_prep_pipe = self.base_pipe(num_columns=self.num_columns, cat_columns=self.cat_columns)
        self.model_fit_cv(strat=strat,groups=groups)
        self.mean_metric_folds = list()
        for i in range(1,len(self._main_scores.keys())+1):
            self.mean_metric_folds.append(self._main_scores[f'scores_{i}']['main_valid'])
        self.mean_metric_folds = np.array(self.mean_metric_folds)
        fi = self.get_fi()
        features_imp = fi[fi['mean_importance']>0]['index'].to_list()
        
        bselection_res = dict()
        bselection_res['features_stack'] = None
        
        print('Средняя метрика на фолдах со всеми факторами: ',np.round(np.mean(self.mean_metric_folds),6))
        print('==========================')
        print('Конец обуения кросс-валидации!')
        print()
        
        print('Начало жадного отбора факторов Backward Selection!')
        print('==========================')

        features_stack = features_imp
        features_drop = list()
        features_scores = list()
        for i in range(len(features_stack) - 1):
            features_drop.append(features_stack[-1])
            features_stack = features_stack[:-1]
            _fold_scores = list()
            print(f'Удаление признака {i}')
            
            num_stack = list(filter(lambda x: x in self.num_columns, features_stack))
            cat_stack = list(filter(lambda x: x in self.cat_columns, features_stack))            
            
            self.main_features = num_stack+cat_stack
            
            if len(num_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='cat')
            elif len(cat_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='num')
            else:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='all')
            
            self.model_fit_cv(strat=strat, groups=groups)
            
            _fold_scores = list()
            for i in range(1,len(self._main_scores.keys())+1):
                _fold_scores.append(self._main_scores[f'scores_{i}']['main_valid'])
            _fold_scores = np.array(_fold_scores)
            mean_fold_scores = np.mean(_fold_scores)
            
            features_scores.append(mean_fold_scores)
            print(f'Количество признаков: {len(features_stack)} => метрика: {mean_fold_scores}')
            
        
            # Условие стопа (если alt лучше на более, чем половине фолдов)
            if self.main_metric in ['accuracy','roc_auc', 'gini', 'delta_gini', 'roc_auc_ovr', 'roc_auc_ovo', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_weighted'] and first_degradation==True:
                if (np.sum(_fold_scores <= self.mean_metric_folds) > strat.n_splits // 2):
                    
                    features_stack.append(features_drop[-1])
                    features_drop.pop()
                    
                    bselection_res['features_stack'] = features_stack
                    bselection_res['features_drop'] = features_drop
                    bselection_res['num_features_stack'] = list(filter(lambda x: x in self.num_columns, features_stack))
                    bselection_res['cat_features_stack'] = list(filter(lambda x: x in self.cat_columns, features_stack))
                    bselection_res['metric_alt'] = np.round(features_scores[-2],4)

                    break    
                else:
                    bselection_res['features_stack'] = features_imp
                    bselection_res['features_drop'] = list()
                    bselection_res['num_features_stack'] = self.num_columns
                    bselection_res['cat_features_stack'] = self.cat_columns
                    bselection_res['metric_alt'] = np.round(np.mean(self.mean_metric_folds), 6)
                    
            elif self.main_metric in ['accuracy','roc_auc', 'gini', 'delta_gini', 'roc_auc_ovr', 'roc_auc_ovo', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_weighted'] and first_degradation==False:
                if (np.sum(_fold_scores >= self.mean_metric_folds) > strat.n_splits // 2):
                    bselection_res['features_stack'] = features_stack
                    bselection_res['features_drop'] = features_drop
                    bselection_res['num_features_stack'] = num_stack
                    bselection_res['cat_features_stack'] = cat_stack
                    bselection_res['metric_alt'] = np.round(mean_fold_scores, 6)

                    break    
                else:
                    bselection_res['features_stack'] = features_imp
                    bselection_res['features_drop'] = list()
                    bselection_res['num_features_stack'] = self.num_columns
                    bselection_res['cat_features_stack'] = self.cat_columns
                    bselection_res['metric_alt'] = np.round(np.mean(self.mean_metric_folds), 6)
                    
            elif self.main_metric in ['mae', 'mse', 'rmse', 'mape'] and first_degradation==True:
                if (np.sum(_fold_scores >= self.mean_metric_folds) > strat.n_splits // 2):

                    features_stack.append(features_drop[-1])
                    features_drop.pop()
                    
                    bselection_res['features_stack'] = features_stack
                    bselection_res['features_drop'] = features_drop
                    bselection_res['num_features_stack'] = list(filter(lambda x: x in self.num_columns, features_stack))
                    bselection_res['cat_features_stack'] = list(filter(lambda x: x in self.cat_columns, features_stack))
                    bselection_res['metric_alt'] = np.round(features_scores[-2],4)

                    break    
                else:
                    bselection_res['features_stack'] = features_imp
                    bselection_res['features_drop'] = list()
                    bselection_res['num_features_stack'] = self.num_columns
                    bselection_res['cat_features_stack'] = self.cat_columns
                    bselection_res['metric_alt'] = np.round(np.mean(self.mean_metric_folds), 6)
                    
            elif self.main_metric in ['mae', 'mse', 'rmse', 'mape'] and first_degradation==False:
                if (np.sum(_fold_scores <= self.mean_metric_folds) > strat.n_splits // 2):

                    bselection_res['features_stack'] = features_stack
                    bselection_res['features_drop'] = features_drop
                    bselection_res['num_features_stack'] = num_stack
                    bselection_res['cat_features_stack'] = cat_stack
                    bselection_res['metric_alt'] = np.round(mean_fold_scores, 6)

                    break    
                else:
                    bselection_res['features_stack'] = features_imp
                    bselection_res['features_drop'] = list()
                    bselection_res['num_features_stack'] = self.num_columns
                    bselection_res['cat_features_stack'] = self.cat_columns
                    bselection_res['metric_alt'] = np.round(np.mean(self.mean_metric_folds), 6)
                
            print('==========================')
        print()
        print(f'Количество отобранных признаков: {len(features_stack)}')
        print(f'Метрика до отбора: {np.round(np.mean(self.mean_metric_folds),4)} => после отбора: {np.round(bselection_res["metric_alt"],4)}')
        print('==========================')
        print(f'Конец жадного отбора факторов Backward Selection!')

        self.bselection_res = bselection_res
        self.backward = True
        
        return bselection_res
    
#    def plot_backward(self, save:bool=False, figsize=(12, 8)):
#        fig = plt.figure(figsize=figsize)
#        plt.ylabel(self.main_metric)
#        plt.title('Обратный последовательный отбор признаков')
#        plt.plot(self.bselection_res['features_drop'], self.bselection_res['features_scores'], linewidth=2, markersize=10, marker='s', linestyle='-', label = 'Метрика последовательного отбора')
#        plt.plot([np.mean(self.mean_metric_folds) for c in self.bselection_res['features_scores']], linewidth=4, linestyle=':', label = f'Среднее значение метрики на кросс-валидации = {np.round(np.mean(self.mean_metric_folds),3)}')
#        plt.legend()
#        plt.xticks(rotation=90)
#        plt.grid()
#        if save:
#            plt.savefig('forward_{}.png'.format(self.main_metric), dpi=300)
#        plt.show()
        
    def deep_backward_selection(self, strat, groups=None, tol:float=0.001): 
        
        self.main_features = self.num_columns+self.cat_columns
        self.main_prep_pipe = self.base_pipe(num_columns=self.num_columns, cat_columns=self.cat_columns)
        self.model_fit_cv(strat=strat,groups=groups)
        self.mean_metric_folds = list()
        for i in range(1,len(self._main_scores.keys())+1):
            self.mean_metric_folds.append(self._main_scores[f'scores_{i}']['main_valid'])
        self.mean_metric_folds = np.array(self.mean_metric_folds)
        mean_metric_folds = np.mean(self.mean_metric_folds)
        fi = self.get_fi()
        features_imp = fi[fi['mean_importance']>0]['index'].to_list()[::-1]
        
        bselection_res = dict()
        bselection_res['features_stack'] = None
        
        print('Средняя метрика на фолдах со всеми факторами: ',np.round(np.mean(self.mean_metric_folds),6))
        print('==========================')
        print('Конец обучения кросс-валидации!')
        print()
        
        print('Начало глубокого жадного отбора факторов Backward Selection!')
        print('==========================')

        features_to_remove = list()
        metric_mean_list = list()
        diff_metric_list = list()
        count = 1
        
        for i in features_imp:
            print(f'Проверяемый признак: {i}')
            count = count + 1
            
            features_stack = self.X_train[features_imp].drop(features_to_remove + [i], axis=1).columns.tolist()[::-1]
            
            num_stack = list(filter(lambda x: x in self.num_columns, features_stack))
            cat_stack = list(filter(lambda x: x in self.cat_columns, features_stack))            
            
            self.main_features = num_stack+cat_stack
            
            if len(num_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='cat')
            elif len(cat_stack)==0:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='num')
            else:
                self.main_prep_pipe = self.base_pipe(num_columns=num_stack,cat_columns=cat_stack,kind='all')
            
            self.model_fit_cv(strat=strat, groups=groups)
            
            _fold_scores = list()
            for j in range(1,len(self._main_scores.keys())+1):
                _fold_scores.append(self._main_scores[f'scores_{j}']['main_valid'])
            _fold_scores = np.array(_fold_scores)
            mean_fold_scores = np.mean(_fold_scores)
            
            metric_mean_list.append(mean_fold_scores)
            print(f'Количество признаков: {len(features_stack)} => метрика: {mean_fold_scores}')
            print('Метрика модели со всеми признаками={}'.format((mean_metric_folds)))
            
            diff_metric = mean_metric_folds - mean_fold_scores
            diff_metric_list.append(diff_metric)
            
            # сравниваем разницу метрики с порогом, заданным заранее
            # если разница метрики больше или равна порогу, сохраняем
            
            if self.main_metric in ['accuracy','roc_auc', 'gini', 'delta_gini', 'roc_auc_ovr', 'roc_auc_ovo', 'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro', 'recall_weighted']:
            
                if diff_metric >= tol:
                    print('Разница метрики ={}'.format(diff_metric))
                    print('Сохраняем: ', i)
                    print
                # если разница метрики меньше порога, удаляем
                else:
                    print('Разница метрики ={}'.format(diff_metric))
                    print('Удаляем: ', i)
                    print

                    # если разница метрики меньше порога и мы удаляем признак,
                    # мы в качестве нового опорного значения метрики задаем
                    # значение метрики для модели с оставшимися признаками
                    mean_metric_folds = mean_fold_scores

                    # добавляем удаляемый признак в список
                    features_to_remove.append(i)
                  
            elif self.main_metric in ['mae', 'mse', 'rmse', 'mape']:
                
                if diff_metric <= tol:
                    print('Разница метрики ={}'.format(diff_metric))
                    print('Сохраняем: ', i)
                    print
                # если разница метрики меньше порога, удаляем
                else:
                    print('Разница метрики ={}'.format(diff_metric))
                    print('Удаляем: ', i)
                    print

                    # если разница метрики меньше порога и мы удаляем признак,
                    # мы в качестве нового опорного значения метрики задаем
                    # значение метрики для модели с оставшимися признаками
                    mean_metric_folds = mean_fold_scores

                    # добавляем удаляемый признак в список
                    features_to_remove.append(i)
        
        # определяем признаки, которые мы хотим сохранить (не удаляем)
        features_to_keep = [x for x in features_imp if x not in features_to_remove]
        num_stack = list(filter(lambda x: x in self.num_columns, features_to_keep))
        cat_stack = list(filter(lambda x: x in self.cat_columns, features_to_keep))
        
        bselection_res['features_stack'] = features_to_keep[::-1]
        bselection_res['features_drop'] = features_to_remove
        bselection_res['num_features_stack'] = num_stack[::-1]
        bselection_res['cat_features_stack'] = cat_stack[::-1]
        bselection_res['metric_alt'] = mean_metric_folds

        print('Общее количество признаков для сохранения: ', len(features_to_keep))
        print()    
        print('==========================')
        print(f'Конец жадного отбора факторов Backward Selection!')

        self.deep_bselection_res = bselection_res
        self.dbackward = True 
        
        return bselection_res        
    
    def report(self, X: pd.DataFrame, y: pd.DataFrame):
        
        method = list()
        cnt_features = list()
        cross_metrics = list()
        oof_metrics = list()
        
        
        # Mean importances

        self.main_features = self.num_columns+self.cat_columns
        self.main_prep_pipe = self.base_pipe(num_columns=self.num_columns, cat_columns=self.cat_columns)
        cnt_fi = len(self.main_features)

        X_train_fi = self.main_prep_pipe.fit_transform(self.X_train[self.main_features], self.y_train)
        model_fi = self.main_estimator.fit(X_train_fi, self.y_train)

        X_test_fi = self.main_prep_pipe.transform(X[self.main_features])        
        y_test_fi = _get_score_selection(X_test_fi,model_fi,self.model_type,self.main_metric)
        metric_fi = np.round(self._get_metric(y,y_test_fi,self.main_metric),4)
        
        method.append('Mean_importance')
        cnt_features.append(cnt_fi)
        cross_metrics.append(np.mean(self.mean_metric_folds))
        oof_metrics.append(metric_fi)        

        # Forward selection
        if self.forward:
            main_features = self.fselection_res['num_features_stack']+self.fselection_res['cat_features_stack']
            self.main_prep_pipe = self.base_pipe(num_columns=self.fselection_res['num_features_stack'], cat_columns=self.fselection_res['cat_features_stack'])
            cnt_forw = len(main_features)

            X_train_forw = self.main_prep_pipe.fit_transform(self.X_train[main_features], self.y_train)
            model_forw = self.main_estimator.fit(X_train_forw, self.y_train)

            X_test_forw = self.main_prep_pipe.transform(X[main_features])        
            y_test_forw = _get_score_selection(X_test_forw,model_forw,self.model_type,self.main_metric)
            metric_forw = np.round(self._get_metric(y,y_test_forw,self.main_metric),4)

            method.append('Forward_selection')
            cnt_features.append(cnt_forw)
            cross_metrics.append(self.fselection_res['metric_alt'])
            oof_metrics.append(metric_forw) 

        # Backward selection
        if self.backward:
            main_features = self.bselection_res['num_features_stack']+self.bselection_res['cat_features_stack']
            self.main_prep_pipe = self.base_pipe(num_columns=self.bselection_res['num_features_stack'], cat_columns=self.bselection_res['cat_features_stack'])
            cnt_back = len(main_features)

            X_train_back = self.main_prep_pipe.fit_transform(self.X_train[main_features], self.y_train)
            model_back = self.main_estimator.fit(X_train_back, self.y_train)

            X_test_back = self.main_prep_pipe.transform(X[main_features])        
            y_test_back = _get_score_selection(X_test_back,model_back,self.model_type,self.main_metric)
            metric_back = np.round(self._get_metric(y,y_test_back,self.main_metric),4)
            
            method.append('Backward_selection')
            cnt_features.append(cnt_back)
            cross_metrics.append(self.bselection_res['metric_alt'])
            oof_metrics.append(metric_back) 

        # Deep backward selection
        if self.dbackward:
            main_features = self.deep_bselection_res['num_features_stack']+self.deep_bselection_res['cat_features_stack']
            self.main_prep_pipe = self.base_pipe(num_columns=self.deep_bselection_res['num_features_stack'], cat_columns=self.deep_bselection_res['cat_features_stack'])
            cnt_dback = len(main_features)

            X_train_dback = self.main_prep_pipe.fit_transform(self.X_train[main_features], self.y_train)
            model_dback = self.main_estimator.fit(X_train_dback, self.y_train)

            X_test_dback = self.main_prep_pipe.transform(X[main_features])        
            y_test_dback = _get_score_selection(X_test_dback,model_dback,self.model_type,self.main_metric)
            metric_dback = np.round(self._get_metric(y,y_test_dback,self.main_metric),4)
            
            method.append('Deep_backward_selection')
            cnt_features.append(cnt_dback)
            cross_metrics.append(self.deep_bselection_res['metric_alt'])
            oof_metrics.append(metric_dback) 

        result_df = pd.DataFrame({'Метод отбора':method,'Количество факторов':cnt_features,f'Метрика {self.main_metric} на кросс-валидации':cross_metrics,f'Метрика {self.main_metric} на отложенном множестве':oof_metrics})
        
        return result_df