# импортируем необходимые библиотеки и классы
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoost, Pool
import xgboost as xgb
import operator
import warnings
import shap

class StochasticBoostARoota(BaseEstimator, TransformerMixin):
    # все параметры для инициализации публичных атрибутов
    # задаем в методе __init__
    def __init__(self, metric=None, selection_algorithm='xgb', model_spec=None, 
                 cutoff=4, iters=10, max_rounds=100, delta=0.1, 
                 frac=1, seed=None, sampling='full', silent=False):
        # оптимизируемая метрика
        self.metric = metric
        # алгоритм на основе деревьев для отбора признаков
        # (по умолчанию XGBoost, clf='xgb')
        self.selection_algorithm = selection_algorithm
        # спецификация модели
        self.model_spec = model_spec
        # порог отсечения для удаления признаков, исходя из их важности. 
        # По умолчанию равен 4. Большие значения будут более консервативными - 
        # если установить значение параметра слишком высоким, в конечном итоге 
        # может быть удалено незначительное количество признаков. Маленькие
        # значения будут более агрессивными. Значение должно быть выше нуля 
        # (может быть значение с плавающей точкой).
        self.cutoff = cutoff
        # количество итераций (запусков алгоритма XGBoost) для усреднения важности
        # признаков. По умолчанию равен 10. Не рекомендуется устанавливать значение 
        # параметра равным 1 (чем меньше итераций, тем выше случайная изменчивость 
        # оценок важностей и быстрее скорость вычислений, так как алгоритм запустит 
        # XGBoost меньшее количество раз).
        self.iters = iters
        # количество раундов работы основного алгоритма BoostARoota. Каждый раунд
        # устраняет все больше и больше признаков. Значение по умолчанию 
        # установлено достаточно высоким, т.к. все равно при нормальных 
        # обстоятельствах такое количество рандов не потребуется. Если вам 
        # кажется, что переменные удаляются слишком агрессивно, вы можете 
        # установить более низкое значение.
        self.max_rounds = max_rounds
        # доля удаляемых признаков для перехода к следующему раунду
        self.delta = delta
        # доля наблюдений, используемых для обучения
        self.frac = frac
        # стартовое значение генератора псевдослучайных чисел
        self.seed = seed
        # вывод сообщений о ходе работы
        self.silent = silent
        # тип отбора наблюдений
        self.sampling = sampling
        # переменные, отобранные алгоритмом
        self.keep_vars_ = None
        
        
        # выдать ошибки, если для параметров заданы некорректные значения
        if (metric is not None) and (selection_algorithm != 'xgb'):
            # сообщение о том, что метрика задается, если не выбран алгоритм отбора,
            raise ValueError("The parameter 'metric' is set explicitly only when\n" +
                             "selection_algorithm='xgb'. If you choose 'catbst'\n" + 
                             "or 'sklearn' you should set metric=None.")
        if cutoff <= 0:
            # сообщение о том, что cutoff должен быть больше 0
            raise ValueError("Cutoff should be greater than 0. You entered" + str(cutoff))
        if iters <= 0:
            # сообщение о том, что iters должен быть больше 0
            raise ValueError("Iters should be greater than 0. You entered" + str(iters))
        if (delta <= 0) | (delta > 1):
            # сообщение о том, что значение delta должно быть больше 0, но не больше 1 
            raise ValueError("Delta should be between 0 and 1, was " + str(delta))
        if (frac < 0.5) | (frac > 1):
            # сообщение о том, что значение frac должно быть не меньше 0.5, но не больше 1 
            raise ValueError("Frac should be between 0.5 and 1, was " + str(frac))
        if (frac < 1) & (sampling == 'full'):
            # сообщение о том, что значение frac не может быть меньше 1 для режима full
            raise ValueError("If you set sampling='full' frac cannot be less than 1, was " + str(frac))
       
                
        if delta < 0.02:
            # предупреждение о том, что при delta < 0.02 алгоритм может не сойтись
            warnings.warn("WARNING: Setting a delta below 0.02 may not converge on a solution.")
        if max_rounds < 1:
            # предупреждение о том, что если значение max_rounds установлено ниже 1, 
            # это значение будет автоматически задано равным 1
            warnings.warn("WARNING: Setting max_rounds below 1 will automatically be set to 1.")
            
    # метод .fit() выполняет обучение - отбор признаков
    def fit(self, x, y):
            
        # вызов основной функции, запускающей алгоритм BoostARoota,
        # возвращаем имена переменных, которые нужно сохранить
        self.keep_vars_ = self._BoostARoota(x, y,
                                            metric=self.metric,
                                            selection_algorithm=self.selection_algorithm,
                                            model_spec=self.model_spec,
                                            cutoff=self.cutoff,
                                            iters=self.iters,
                                            max_rounds=self.max_rounds,
                                            delta=self.delta,
                                            frac=self.frac,
                                            seed=self.seed,
                                            silent=self.silent,
                                            sampling=self.sampling)
        return self

    # метод .transform() формирует новый массив -
    # массив отобранных признаков
    def transform(self, x):
        # если переменная keep_vars_ еще не определена и был вызван метод .transform()
        if self.keep_vars_ is None:
            # выводится сообщение о том, что сначала нужно применить метод .fit()
            raise ValueError("You need to fit the model first")
        # возвращает массив данных с теми признаками, которые
        # были отобраны алгоритмом BoostARoota в методе .fit()
        return x[self.keep_vars_]

    # функция добавления "теневых" признаков. Она удваивает ширину набора данных, создав
    # копии всех признаков исходного набора. Случайным образом перемешивает значения новых 
    # признаков. Эти дублированные и перемешанные признаки называются «теневыми».    
    def _create_shadow(self, x_train, seed):
        """
        :параметр x_train: датафрейм данных для создания 
            на его основе "теневых" признаков
        :возвращает: датафрейм данных удвоенной ширины и имена 
            "теневых" признаков для последующего удаления
        """
        # создаем копию обучающего массива признаков
        x_shadow = x_train.copy()
        # в цикле проходим по всем "теневым" признакам
        # и перемешиваем их значения
        
        if self.seed == None:
            for c in x_shadow.columns:
                np.random.shuffle(x_shadow[c].values)
            
        else:
            np.random.seed(seed**2 + len(x_shadow.columns)**2)
            for c in x_shadow.columns:
                np.random.shuffle(x_shadow[c].values)

        # переименовываем "теневые" признаки
        shadow_names = ["ShadowVar" + str(i + 1) for i in range(x_train.shape[1])]
        x_shadow.columns = shadow_names
        # объединяем набор из исходных и набор из "теневых" признаков 
        # в один новый датафрейм удвоенной ширины
        new_x = pd.concat([x_train, x_shadow], axis=1)
        # возвращаем датафрейм удвоенной ширины из исходных и "теневых" признаков
        # и список имен "теневых" признаков для последующего удаления
        return new_x, shadow_names

    # функция вычисления важностей для отбора признаков 
    # на основе алгоритма XGBoost
    def _reduce_vars_xgb(self, x, y, metric, selection_algorithm, this_round, cutoff, n_iterations, 
                         delta, frac, seed, silent, sampling):
        """
        :параметр x: входной массив признаков - X
        :параметр y: зависимая переменная
        :параметр metric: оптимизируемая метрика в XGBoost
        :параметр this_round: номер текущего раунда, чтобы его можно было вывести на экран
        :возвращает: кортеж - критерий остановки и имена переменных, которые нужно сохранить
        """
        # если метрика для оптимизации - mlogloss, то задаем соответствующую
        # функцию потерь, оптимизируемую метрику, количество классов, "тихий"
        # режим для обучения модели XGBoost 
        if metric == 'mlogloss':
            param = {'objective': 'multi:softmax',
                     'eval_metric': 'mlogloss',
                     'num_class': len(np.unique(y)),
                     'verbosity': 0}
        else:
            # в противном случае оптимизируемой метрикой 
            # будет заданная метрика для оптимизации
            param = {'eval_metric': metric,
                     'verbosity': 0}

        # выполнение в цикле итераций обучения алгоритма XGBoost 
        # для усреднения важности признаков
        for i in range(1, n_iterations + 1):

            if sampling == 'random':
                # выполняем случайный отбор наблюдений
                np.random.seed((seed + i)**2)
                idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                xs = np.take(x, idx, axis=0)
                ys = np.take(y, idx, axis=0)
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(xs, seed + i)
                # преобразовываем массив признаков и массив меток в объект DMatrix 
                dtrain = xgb.DMatrix(new_x, label=ys)

            if sampling == 'reduced_random':
                # выполняем уменьшающийся случайный отбор наблюдений
                
                if self.seed == None:
                    idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                else:
                    np.random.seed((seed + i)**2)
                    idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                    
                x = np.take(x, idx, axis=0)
                y = np.take(y, idx, axis=0) 
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                
                if self.seed == None:
                    new_x, shadow_names = self._create_shadow(x, seed)
                else:
                    new_x, shadow_names = self._create_shadow(x, seed + i)
                # преобразовываем массив признаков и массив меток в объект DMatrix 
                dtrain = xgb.DMatrix(new_x, label=y)

            if sampling == 'full':
                # используем весь набор наблюдений
                
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(x, seed + i)
                # преобразовываем массив признаков и массив меток в объект DMatrix 
                dtrain = xgb.DMatrix(new_x, label=y)

            # обучаем модель XGBoost
            
            if self.seed == None:
                pass
            else:
                param.update({'seed': (seed + i)**2})
            
            bst = xgb.train(param, dtrain, verbose_eval=False)
            # если это первая итерация
            if i == 1:
                # создаем датафрейм df со столбцом-списком признаков
                df = pd.DataFrame({'feature': new_x.columns})
                pass

            # получаем значение важности для каждого признака, по умолчанию
            # используется weight - простой показатель важности, который 
            # суммирует, сколько раз конкретный признак использовался 
            # в качестве предиктора разбиения в алгоритме XGBoost      
            importance = bst.get_fscore()
            # сортируем по значению важности
            importance = sorted(importance.items(), key=operator.itemgetter(1))
            # создаем датафрейм, содержащий названия предикторов и их важности
            df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
            # нормируем значения важности
            df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
            # объединяем датафреймы df и df2, т.е. к df добавляется столбец со 
            # значениями важности, найденными в текущей i-ой итерации
            df = pd.merge(df, df2, on='feature', how='outer')
            # если не задан "тихий" режим, печатаем информацию
            # о текущем раунде и итерации
            if not silent:
                print("Round: ", this_round, " iteration: ", i)
                
            if len(new_x) < 60:
                break

        # в df добавляем усредненное значение важности по всем пройденным итерациям 
        df['Mean'] = df.mean(axis=1)
        # выполняем обратное разделение признаков на исходные и "теневые"
        real_vars = df[~df['feature'].isin(shadow_names)]
        shadow_vars = df[df['feature'].isin(shadow_names)]

        # вычисляем «порог отсечения»: среднее значение важности 
        # для всех «теневых» признаков, поделенное на значение 
        # cutoff (по умолчанию равно 4)
        mean_shadow = shadow_vars['Mean'].mean() / cutoff

        # удаляем признаки, средняя важность которых по результатам 
        # всех итераций меньше «порога отсечения»
        real_vars = real_vars[(real_vars.Mean > mean_shadow)]

        # проверяем критерий остановки
        # в основном мы хотим убедиться, что удаляем не менее 10% переменных, 
        # иначе следует остановиться
        if (len(real_vars['feature']) / len(x.columns)) > (1 - delta):
            criteria = True
        else:
            criteria = False

        # возвращаем критерий остановки и список оставшихся признаков
        return criteria, real_vars['feature']

    def _reduce_vars_catbst(self, x, y, selection_algorithm, model_spec, this_round, cutoff, 
                            n_iterations, delta, frac, seed, silent, sampling):  
        """
        :параметр x: входной массив признаков - X
        :параметр y: зависимая переменная
        :параметр clf: алгоритм из библиотеки sklearn на основе 
            деревьев решений, переданный пользователем
        :параметр this_round: номер текущего раунда, чтобы его можно было вывести на экран
        :возвращает: кортеж - критерий остановки и имена переменных, которые нужно сохранить    
        """

        # выполнение в цикле итераций обучения указанного алгоритма
        # для усреднения важности признаков
        for i in range(1, n_iterations + 1):
            
            if sampling == 'random':
                # выполняем случайный отбор наблюдений
                np.random.seed((seed + i)**2)
                idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                xs = np.take(x, idx, axis=0)
                ys = np.take(y, idx, axis=0)
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(xs, seed + i)
                categorical_features_indices = np.where(new_x.dtypes != np.float)[0] 
                train_pool = Pool(new_x, ys, cat_features=categorical_features_indices)

            if sampling == 'reduced_random':
                # выполняем уменьшающийся случайный отбор наблюдений
                np.random.seed((seed + i)**2)
                idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                x = np.take(x, idx, axis=0)
                y = np.take(y, idx, axis=0) 
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(x, seed + i)
                categorical_features_indices = np.where(new_x.dtypes != np.float)[0] 
                train_pool = Pool(new_x, y, cat_features=categorical_features_indices)

            if sampling == 'full':
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(x, seed + i)
                categorical_features_indices = np.where(new_x.dtypes != np.float)[0] 
                train_pool = Pool(new_x, y, cat_features=categorical_features_indices)

            parameters = model_spec.get_params()
            parameters.update({'random_state': (seed + i)**2})
            model_spec = CatBoost(params=parameters)
            model_spec.fit(train_pool)
            print(model_spec.get_params())

            # если это первая итерация
            if i == 1:
                # создаем датафрейм df со столбцом-списком признаков
                df = pd.DataFrame({'feature': new_x.columns})
                pass

            # вычисляем важности на основе значений SHAP
            shap_values = model_spec.get_feature_importance(train_pool, 'ShapValues')
            # удаляем базовое значение
            shap_values = shap_values[:, :-1]
            # задаем список важностей
            values = np.abs(shap_values).mean(0).tolist()
            # задаем список предикторов
            features = new_x.columns.tolist()
            # задаем список двухэлементных кортежей
            importance = list(zip(features, values))
            # создаем датафрейм, содержащий названия предикторов и их важности
            df2 = pd.DataFrame(importance, columns=['feature', 'fscore'+str(i)])
            # нормируем значения важности
            df2['fscore' + str(i)] = df2['fscore' + str(i)] / df2['fscore' + str(i)].sum()
            # объединяем датафреймы df и df2, т.е. к df добавляется столбец со 
            # значениями важности, найденными в текущей i-ой итерации
            df = pd.merge(df, df2, on='feature', how='outer')  
            # если не задан "тихий" режим, печатаем информацию
            # о текущем раунде и итерации
            if not silent:
                print("Round: ", this_round, " iteration: ", i)
                
            if len(new_x) < 500:
                break

        # в df добавляем усредненное значение важности по всем пройденным итерациям
        df['Mean'] = df.mean(axis=1)
        # выполняем обратное разделение признаков на исходные и "теневые"
        real_vars = df[~df['feature'].isin(shadow_names)]
        shadow_vars = df[df['feature'].isin(shadow_names)]

        # вычисляем «порог отсечения»: среднее значение важности 
        # для всех «теневых» признаков, поделенное на значение 
        # cutoff (по умолчанию равно 4)
        mean_shadow = shadow_vars['Mean'].mean() / cutoff

        # удаляем признаки, средняя важность которых по результатам 
        # всех итераций меньше «порога отсечения»
        real_vars = real_vars[(real_vars.Mean > mean_shadow)]

        # проверяем критерий остановки
        # в основном мы хотим убедиться, что удаляем не менее 10% переменных, 
        # иначе следует остановиться
        if (len(real_vars['feature']) / len(x.columns)) > (1 - delta):
            criteria = True
        else:
            criteria = False

        # возвращаем критерий остановки и список оставшихся признаков
        return criteria, real_vars['feature']

    def _reduce_vars_sklearn(self, x, y, selection_algorithm, model_spec, this_round, cutoff, 
                             n_iterations, delta, frac, seed, silent, sampling):
        
        # выполнение в цикле итераций обучения указанного алгоритма
        # для усреднения важности признаков
        for i in range(1, n_iterations+1):

            if sampling == 'random':
                # выполняем случайный отбор наблюдений
                # np.random.seed((seed + i)**2)
                idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                xs = np.take(x, idx, axis=0)
                ys = np.take(y, idx, axis=0)
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(xs, seed + i)
                # задали обучение модели sklearn
                spec = spec.fit(new_x, np.ravel(ys))

            if sampling == 'reduced_random':
                # выполняем уменьшающийся случайный отбор наблюдений
                np.random.seed((seed + i)**2)
                idx = np.random.choice(x.shape[0], int(frac * len(x)), replace=False)
                x = np.take(x, idx, axis=0)
                y = np.take(y, idx, axis=0)
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(x, seed + i)
                # задали обучение модели sklearn
                spec = spec.fit(new_x, np.ravel(y))

            if sampling == 'full':
                # используем весь набор наблюдений
                    
                # создаем "теневые" признаки:
                # new_x - содержит датафрейм удвоенной ширины 
                # с исходными и "теневыми" предикторами
                # shadow_names - список имен "теневых" признаков 
                # для последующего удаления
                new_x, shadow_names = self._create_shadow(x, seed + i)
                # задали обучение модели sklearn
                spec = spec.fit(new_x, np.ravel(y))

            # если это первая итерация
            if i == 1:
                # создаем датафрейм df со столбцом-списком признаков
                df = pd.DataFrame({'feature': new_x.columns})
                # копируем его в датафрейм df2
                df2 = df.copy()
                pass

            try:
                # получаем значение важности для каждого признака
                # с помощью атрибута feature_importances_
                importance = spec.feature_importances_
                # добавляем в df2 значения важности, найденные в текущей итерации
                df2['fscore' + str(i)] = importance
            except ValueError:
                # выдается ошибка, если задан алгоритм, в котором 
                # нет атрибута feature_importances_
                print("this clf doesn't have the feature_importances_ method.\n" +
                      "Only Sklearn tree based methods allowed")

            # нормируем значения важности
            df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
            # объединяем датафреймы df и df2, т.е. к df добавляется столбец со 
            # значениями важности, найденными в текущей i-ой итерации
            df = pd.merge(df, df2, on='feature', how='outer')
            # если не задан "тихий" режим, печатаем информацию
            # о текущем раунде и итерации
            if not silent:
                print("Round: ", this_round, " iteration: ", i)
                
            if len(new_x) < 500:
                break
        
        # в df добавляем усредненное значение важности по всем пройденным итерациям         
        df['Mean'] = df.mean(axis=1)
        # выполняем обратное разделение признаков на исходные и "теневые"
        real_vars = df[~df['feature'].isin(shadow_names)]
        shadow_vars = df[df['feature'].isin(shadow_names)]

        # вычисляем «порог отсечения»: среднее значение важности 
        # для всех «теневых» признаков, поделенное на значение 
        # cutoff (по умолчанию равно 4)
        mean_shadow = shadow_vars['Mean'].mean() / cutoff

        # удаляем признаки, средняя важность которых по результатам 
        # всех итераций меньше «порога отсечения»
        real_vars = real_vars[(real_vars.Mean > mean_shadow)]
        
        # проверяем критерий остановки
        # в основном мы хотим убедиться, что удаляем не менее 10% переменных, 
        # иначе следует остановиться
        if (len(real_vars['feature']) / len(x.columns)) > (1 - delta):
            criteria = True
        else:
            criteria = False

        # возвращаем критерий остановки и список оставшихся признаков
        return criteria, real_vars['feature']
            

    # основная функция, запускающая алгоритм BoostARoota
    def _BoostARoota(self, x, y, metric, selection_algorithm, model_spec, cutoff, iters, 
                     max_rounds, delta, frac, seed, silent, sampling):
        """
        Функция проходит цикл, ожидая изменения критерия остановки
        :параметр x: массив признаков X (если есть категориальные переменные,
            нужно выполнить дамми-кодирование)
        :параметр y: массив меток зависимой переменной
        :параметр metric: оптимизируемая метрика
        :возвращает: имена переменных, которые нужно сохранить
        """
        # создаем копию массива признаков
        new_x = x.copy()
        num_of_var = len(x.columns)

        # выполняем цикл до тех пор, пока переменная crit не изменится
        # выставляем в ноль счетчик раундов
        i = 0
        while True: # внутри этого цикла мы уменьшаем набор данных на каждом раунде
            # увеличиваем счетчик раундов
            i += 1
            # модифицируем стартовое значение генератора псевдослучайных чисел
            if self.seed == None:
                rnd = None
            else:
                rnd = seed + i * num_of_var
            
            if selection_algorithm == 'xgb':
                # вызывается функция _reduce_vars_xgb, которая возвращает критерий 
                # остановки и уменьшенный на данном раунде список предикторов
                crit, keep_vars = self._reduce_vars_xgb(new_x,
                                                        y,
                                                        metric=metric,
                                                        selection_algorithm=selection_algorithm,
                                                        this_round=i,
                                                        cutoff=cutoff,
                                                        n_iterations=iters,
                                                        delta=delta,
                                                        frac=frac,
                                                        seed=rnd,
                                                        silent=silent,
                                                        sampling=sampling)
            
            if selection_algorithm == 'sklearn':
                # вызывается функция _reduce_vars_sklearn (алгоритм передается
                # через параметр clf), которая возвращает критерий остановки
                # и уменьшенный на данном раунде список предикторов
                crit, keep_vars = self._reduce_vars_sklearn(new_x,
                                                            y,
                                                            selection_algorithm=selection_algorithm,
                                                            model_spec=model_spec,
                                                            this_round=i,
                                                            cutoff=cutoff,
                                                            n_iterations=iters,
                                                            delta=delta,
                                                            frac=frac,
                                                            seed=rnd,
                                                            silent=silent,
                                                            sampling=sampling)
                
            if selection_algorithm == 'catbst':
                # вызывается функция _reduce_vars_catbst (алгоритм передается
                # через параметр clf), которая возвращает критерий остановки
                # и уменьшенный на данном раунде список предикторов
                crit, keep_vars = self._reduce_vars_catbst(new_x,
                                                           y,
                                                           selection_algorithm=selection_algorithm,
                                                           model_spec=model_spec,
                                                           this_round=i,
                                                           cutoff=cutoff,
                                                           n_iterations=iters,
                                                           delta=delta,
                                                           frac=frac,
                                                           seed=rnd,
                                                           silent=silent,
                                                           sampling=sampling)    
                
                
            # если критерий остановки принял значение True 
            # или достигнуто максимальное количество раундов
            if crit | (i >= max_rounds):
                break
                # то выйти из цикла и использовать keep_vars в качестве
                # итогового списка отобранных переменных
            # в противном случае
            else:
                # массив признаков уменьшается, остаются только признаки из keep_vars
                new_x = new_x[keep_vars].copy()
        # если режим не является "тихим"
        if not silent:
            # напечатать сообщение об успешном завершении работы алгоритма
            print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
        return keep_vars