import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from autobinary.trees import AutoTrees


class TargetPermutationSelection(AutoTrees):
    """
    Описание: Класс TargetPermutationSelection пердназначен для отбора признаков с помощью перемешивания таргета
    и оценки предсказательной способности фичей.

    Последовательность действий алгоритма:
    1) Происходит обучение модели с оригинальным тарегетом (benchmark-модель);
    2) Происходит расчет метрики benchmark-модели;
    3) Происходит перемешивание таргета;
    4) Происходит обучение модели с перемешаным таргетом (target permutation-модель);
    5) Происходит расчет метрики target permutation-модели;
    6) Расчет разности метрик 2) и 5)
    7) Происходит отбор признаков   

    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.DataFrame, main_metric: str,
                 main_estimator: object, main_fit_params: dict, main_prep_pipe: object, main_features: list,
                 model_type: str = 'xgboost',
                 extra_estimator: object = None, extra_fit_params: dict = None, extra_prep_pipe: object = None,
                 extra_features: list = None,
                 solo_model: bool = False, treatment: pd.DataFrame = None, uplift: str = None):
        super().__init__(X_train, y_train, main_metric, main_estimator, main_fit_params, main_prep_pipe, main_features,
                         model_type, extra_estimator, extra_fit_params, extra_prep_pipe, extra_features, solo_model,
                         treatment, uplift)
        self._best_iters = []
        self._fi = []
        self.selected_features = None
        self.df_fi_final = pd.DataFrame()
        self.df_permutation_fi = pd.DataFrame()
        self.df_benchmark_fi = pd.DataFrame()

    @staticmethod
    def block_print() -> None:
        sys.stdout = open(os.devnull, 'w')

    def _preprocessing(self, X_tr, X_val, y_tr, y_val, main):
        """Применение конвейера предобработки для не аплифт модели.

        Args:
            X_tr (_type_): _description_
            X_val (_type_): _description_
            y_tr (_type_): _description_
            y_val (_type_): _description_

        Returns:
            _type_: _description_
        """
        if main == 'main':
            self.main_prep_pipe.fit(X_tr, y_tr)

            X_tr = self.main_prep_pipe.transform(X_tr).reset_index(drop=True)
            X_val = self.main_prep_pipe.transform(X_val).reset_index(drop=True)

        elif main == 'extra':
            self.extra_prep_pipe.fit(X_tr, y_tr)

            X_tr = self.extra_prep_pipe.transform(X_tr).reset_index(drop=True)
            X_val = self.extra_prep_pipe.transform(X_val).reset_index(drop=True)

        X_tr = X_tr.assign(random_feature=np.random.randn(len(X_tr)))
        X_val = X_val.assign(random_feature=np.random.randn(len(X_val)))

        y_tr = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        return X_tr, X_val, y_tr, y_val


    def calculate_permutation(self, test_size: float = 0.3) -> pd.DataFrame:
        """
        Описание: функция calculate_permutation предназначена для расчета target permutation importance.
        random_feature - создание случайного признака для отбора из существующих на тестовом множестве True / False.
        Parameters
        ----------
        strat - метод стратификации стратификации (sklearn)

        Returns
        -------
        df_fi_final - финальный датафрейм с важностями фичей

        """
        self.main_features += ['random_feature']
        self.X_train['random_feature'] = np.random.randn(len(self.X_train))
        x_tr, x_te, y_tr, y_te = train_test_split(self.X_train,
                                                  self.y_train,
                                                  test_size=test_size,
                                                  random_state=2022)
        # обучение на кросс-валидации
        print('============================== Обучение benchmark-модели! ==============================')
        best_iter, imp, metr, evals, main_metric = self._model_fit(
            X_train=x_tr,
            y_train=y_tr,
            X_val=x_te,
            y_val=y_te,
            trt_train=None,
            trt_val=None)
        # получение таблицы важностей факторов
        self._fi = []
        self._fi.append(imp)
        self.df_benchmark_fi = self.get_fi()[['index', 'mean_importance']]# .sort_values(by=['index'])
        self.df_benchmark_fi = self.df_benchmark_fi.rename(columns={'mean_importance': 'importance_benchmark'})
        # вывод метрики
        print(f'{self.main_metric} ' + 'на обучающей выборке: {:.3f}'.format(main_metric['main_train']))
        print(f'{self.main_metric} ' + 'на проверочной выборке: {:.3f}'.format(main_metric['main_valid']))

        # перемешивание таргета
        idx = np.arange(len(self.y_train))
        np.random.shuffle(idx)
        self.y_train = self.y_train.iloc[idx]
        x_tr, x_te, y_tr, y_te = train_test_split(self.X_train,
                                                  self.y_train,
                                                  test_size=0.3,
                                                  random_state=2022)
        # кросс-валидация с перешанным таргетом
        print('============================== Обучение target permutation-модели! ==============================')
        best_iter, imp, metr, evals, main_metric = self._model_fit(
            X_train=x_tr,
            y_train=y_tr,
            X_val=x_te,
            y_val=y_te,
            trt_train=None,
            trt_val=None)
        # таблица важностей для модели с перемешанным таргетом
        self._fi = []
        self._fi.append(imp)
        self.df_permutation_fi = self.get_fi()[['index', 'mean_importance']]#.sort_values(by=['index'])
        self.df_permutation_fi = self.df_permutation_fi.rename(columns={'mean_importance': 'importance_permut'})
        # вывод метрики
        print(f'{self.main_metric} ' + 'на обучающей выборке: {:.3f}'.format(main_metric['main_train']))
        print(f'{self.main_metric} ' + 'на проверочной выборке: {:.3f}'.format(main_metric['main_valid']))

        # финальная таблица важностей фичей
        self.df_fi_final['index'] = self.df_benchmark_fi['index']
        self.df_fi_final = self.df_fi_final.join(self.df_benchmark_fi.set_index('index'), on='index')
        self.df_fi_final = self.df_fi_final.join(self.df_permutation_fi.set_index('index'), on='index')
        # self.df_fi_final['importance_benchmark'] = self.df_benchmark_fi['importance']
        # self.df_fi_final['importance_permut'] = self.df_permutation_fi['importance']
        self.df_fi_final['final_importance'] = self.df_fi_final['importance_benchmark'] - self.df_fi_final['importance_permut']
        self.df_fi_final = self.df_fi_final.sort_values(by=['final_importance'], ascending=False)

        return self.df_fi_final


    def select_features(self, select_type: str = 'random_feature', cut_off: float = 0.0) -> dict:
        """
        Описание: функция select_features предназначена для отбора фичей с помощью random_feature.

        Parameters
        ----------
        select_type - флаг для отбора с помощью random_feature True / False

        Returns
        -------
        selected_features - фичи важность, которых выше чем у random_feature
        """
        if select_type == 'random_feature':
            random_score = self.df_fi_final.set_index('index').loc['random_feature'].final_importance
            self.selected_features = self.df_fi_final[self.df_fi_final.final_importance > random_score]['index'].tolist()
        else:
            self.df_fi_final = self.df_fi_final[self.df_fi_final['index'] != 'random_feature']
            self.selected_features = self.df_fi_final[self.df_fi_final.final_importance >= cut_off]['index'].tolist()


        print(len(self.main_features) - 1, 'признаков было до Target Permutation')
        print(len(self.selected_features), 'признаков после Target Permutation')

        return self.selected_features
