# импортируем необходимые библиотеки, классы и функции
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import xgboost as xgb
import operator
import warnings

# создаем собственный класс, заменяющий категории 
# относительными или абсолютными частотами
class CountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=0.1, encoding_method='frequency', nan_value=-1, copy=True):
        
        # если задано значение, не входящее в список, для encoding_method, выдать ошибку
        if encoding_method not in ['count', 'frequency']:
            raise ValueError("encoding_method takes only values 'count' and 'frequency'")
            
        self.min_count = min_count
        self.encoding_method = encoding_method
        self.nan_value = nan_value
        self.copy = copy
        self.counts = {}
                
    def __is_numpy(self, x):
        return isinstance(x, np.ndarray)

    def fit(self, x, y=None):
        # создаем пустой словарь counts
        self.counts = {}
        
        # записываем результат функции is_numpy()
        is_np = self.__is_numpy(x)
        
        # записываем общее количество наблюдений
        n_obs = np.float(len(x))
        
        # если 1D-массив, то переводим в 2D
        if len(x.shape) == 1:
            if is_np:
                x = x.reshape(-1, 1)
            else:
                x = x.to_frame()
            
        # записываем количество столбцов
        ncols = x.shape[1]
    
        for i in range(ncols):
            # если выбрано значение frequency для encoding_method 
            if self.encoding_method == 'frequency':
                # если объект - массив NumPy, выполняем следующие действия:
                if is_np:
                    # создаем временный словарь cnt, ключи - категории переменной,
                    # значения - абсолютные частоты категорий
                    cnt = dict(Counter(x[:, i]))
                    # абсолютные частоты заменяем на относительные
                    cnt = {key: value / n_obs for key, value in cnt.items()}
                # в противном случае, т.е. если объект - Dataframe pandas,
                # выполняем следующие действия:
                else:
                    # создаем временный словарь cnt, ключи - категории переменной,
                    # значения - относительные частоты категорий
                    cnt = (x.iloc[:, i].value_counts() / n_obs).to_dict()
                # если относительная частота категории меньше min_count,
                # возвращаем nan_value
                if self.min_count > 0:
                    cnt = dict((k, self.nan_value if v < self.min_count else v) for k, v in cnt.items())
                    
            # если выбрано значение count для encoding_method        
            elif self.encoding_method == 'count':
                # если объект - массив NumPy, выполняем следующие действия:
                if is_np:
                    # создаем временный словарь cnt, ключи - категории переменной,
                    # значения - абсолютные частоты категорий
                    cnt = dict(Counter(x[:, i]))
                # в противном случае, т.е. если объект - Dataframe pandas,
                # выполняем следующие действия:
                else:
                    # создаем временный словарь cnt, ключи - категории переменной,
                    # значения - абсолютные частоты категорий
                    cnt = (x.iloc[:, i].value_counts()).to_dict()
                # если относительная частота категории меньше min_count,
                # возвращаем nan_value
                if self.min_count > 0:
                    cnt = dict((k, self.nan_value if v < self.min_count else v) for k, v in cnt.items())
                    
            # обновляем словарь counts, ключом словаря counts будет 
            # индекс переменной, значением словаря counts будет 
            # словарь cnt, ключами будут категории переменной, 
            # значениями - относительные частоты переменной
            self.counts.update({i: cnt})
        return self

    def transform(self, x):        
        # выполняем копирование массива
        if self.copy:
            x = x.copy()
            
        # записываем результат функции is_numpy()
        is_np = self.__is_numpy(x)
        
        # если 1D-массив, то переводим в 2D
        if len(x.shape) == 1:
            if is_np:
                x = x.reshape(-1, 1)
            else:
                x = x.to_frame()
                
        # записываем количество столбцов
        ncols = x.shape[1]

        for i in range(ncols):
            cnt = self.counts[i]
            # если объект - массив NumPy, выполняем следующие действия:
            if is_np:
                # получаем из словаря по каждой переменной массив 
                # с категориями и массив с относительными 
                # частотами (в виде строковых значений)
                k, v = np.array(list(zip(*sorted(cnt.items()))))
                # переводим строковые значения в числа с плавающей точкой
                v = np.array(v, float)
                # печатаем индексы, функция np.searchsorted возвращает индексы, 
                # в которые должны быть вставлены указанные элементы, чтобы 
                # порядок сортировки был сохранен, первый параметр - одномерный 
                # исходный массив (массив может быть и не отсортирован, но индексы 
                # возвращаются именно для отсортированной версии), второй параметр - 
                # элементы, которые необходимо вставить в массив
                ix = np.searchsorted(k, x[:, i], side='left')
                # заменяем категории частотами с помощью индексов
                x[:, i] = v[ix]
            # в противном случае, т.е. если объект - Dataframe pandas,
            # выполняем следующие действия:
            else:
                # заменяем категории частотами
                x.iloc[:, i].replace(cnt, inplace=True)
        return x
            
def catboost_target_encoder(train, test, cols_encode, target):
    """
    Encoding based on ordering principle
    """
    train_new = train.copy()
    test_new = test.copy()
    for column in cols_encode:
        global_mean = train[target].mean()
        cumulative_sum = train.groupby(column)[target].cumsum() - train[target]
        cumulative_count = train.groupby(column).cumcount()
        train_new[column + "_cat_mean_enc"] = cumulative_sum/cumulative_count
        train_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(column).mean()[column + "_cat_mean_enc"]
        test_new[column + "_cat_mean_enc"] = test[column].map(col_mean)
        test_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
    # filtering only mean enc cols
    train_new = train_new.filter(like="cat_mean_enc", axis=1)
    test_new = test_new.filter(like="cat_mean_enc", axis=1)
    return train_new, test_new

def one_hot_encoder(train, test, cols_encode, target=None):
    """ one hot encoding"""
    ohc_enc = OneHotEncoder(handle_unknown='ignore')
    ohc_enc.fit(train[cols_encode])
    train_ohc = ohc_enc.transform(train[cols_encode])
    test_ohc = ohc_enc.transform(test[cols_encode])
    return train_ohc, test_ohc

def label_encoder(train, test, cols_encode=None, target=None):
    """
    Code borrowed from fast.ai and is tweaked a little.
    Convert columns in a training and test dataframe into numeric labels 
    """
    train_new = train.drop(target, axis=1).copy()
    test_new = test.drop(target, axis=1).copy()
    
    for n,c in train_new.items():
        if is_string_dtype(c) or n in cols_encode : train_new[n] = c.astype('category').cat.as_ordered()
    
    if test_new is not None:
        for n,c in test_new.items():
            if (n in train_new.columns) and (train_new[n].dtype.name=='category'):
                test_new[n] = pd.Categorical(c, categories=train_new[n].cat.categories, ordered=True)
            
    cols = list(train_new.columns[train_new.dtypes == 'category'])
    for c in cols:
        train_new[c] = train_new[c].astype('category').cat.codes
        if test_new is not None: test_new[c] = test_new[c].astype('category').cat.codes
    return train_new, test_new


def fitmodel_and_auc_score(encoder, train, test, cols_encode, target, **kwargs):
    """
    Fits and returns scores of a gradient boosting model. Uses ROCAUC as scoring metric
    """
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)
    if encoder:
        train_encoder, test_encoder = encoder(train, test, cols_encode=cols_encode, target=target)
    else:
        train_encoder, test_encoder = train.drop(target, axis=1), test.drop(target, axis=1)
    model.fit(train_encoder, train[target])
    train_score = roc_auc_score(train[target], model.predict(train_encoder))
    test_score = roc_auc_score(test[target], model.predict(test_encoder))
    return train_score, test_score

# для работы класса внутри конвейеров потребуется BaseEstimator
class BoostARoota(BaseEstimator, TransformerMixin):
    # все параметры для инициализации публичных атрибутов
    # задаем в методе __init__
    def __init__(self, metric=None, clf=None, cutoff=4, iters=10, 
                 max_rounds=100, delta=0.1, silent=False):
        # оптимизируемая метрика
        self.metric = metric
        # алгоритм на основе деревьев для отбора признаков
        # (по умолчанию XGBoost, clf=None)
        self.clf = clf
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
        # XGBoost меньшее количество раз). Параметр масштабируется линейно: при iters=4 
        # требуется в 2 раза больше времени, чем при iters=2, и в 4 раза больше времени, 
        # чем при iters=1.
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
        # вывод сообщений о ходе работы
        self.silent = silent
        # переменные, отобранные алгоритмом
        self.keep_vars_ = None
        
        # выдать ошибки, если для параметров заданы некорректные значения
        if (metric is None) and (clf is None):
            # сообщение о том, что нужно задать либо метрику, либо алгоритм отбора 
            raise ValueError("you must enter one of metric or clf as arguments")
        if cutoff <= 0:
            # сообщение о том, что cutoff должен быть больше 0
            raise ValueError("cutoff should be greater than 0. You entered" + str(cutoff))
        if iters <= 0:
            # сообщение о том, что iters должен быть больше 0
            raise ValueError("iters should be greater than 0. You entered" + str(iters))
        if (delta <= 0) | (delta > 1):
            # сообщение о том, что значение delta должно быть больше 0, но не больше 1 
            raise ValueError("delta should be between 0 and 1, was " + str(delta))

        # выдать предупреждения при измененных параметрах
        if (metric is not None) and (clf is not None):
            # предупреждение о том, что изменены метрика и алгоритм отбора
            warnings.warn("You entered values for metric and clf, defaulting to clf and ignoring metric")
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
        self.keep_vars_ = _BoostARoota(x, y,
                                       metric=self.metric,
                                       clf=self.clf,
                                       cutoff=self.cutoff,
                                       iters=self.iters,
                                       max_rounds=self.max_rounds,
                                       delta=self.delta,
                                       silent=self.silent)
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
def _create_shadow(x_train):
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
def _reduce_vars_xgb(x, y, metric, this_round, cutoff, n_iterations, delta, silent):
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
        # создаем "теневые" признаки:
        # new_x - содержит датафрейм удвоенной ширины 
        # с исходными и "теневыми" предикторами
        # shadow_names - список имен "теневых" признаков 
        # для последующего удаления
        new_x, shadow_names = _create_shadow(x)
        # преобразовываем массив признаков и массив меток в объект DMatrix 
        dtrain = xgb.DMatrix(new_x, label=y)
        # обучаем модель XGBoost
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

# функция вычисления важностей для отбора признаков на основе алгоритма из 
# библиотеки sklearn, в котором поддерживается атрибут feature_importances_
def _reduce_vars_sklearn(x, y, clf, this_round, cutoff, n_iterations, delta, silent):  
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
    for i in range(1, n_iterations+1):
        # создаем "теневые" признаки:
        # new_x - содержит датафрейм удвоенной ширины 
        # с исходными и "теневыми" предикторами
        # shadow_names - список имен "теневых" признаков 
        # для последующего удаления
        new_x, shadow_names = _create_shadow(x)
        # задали обучение модели sklearn
        clf = clf.fit(new_x, np.ravel(y))
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
            importance = clf.feature_importances_
            # добавляем в df2 значения важности, найденные в текущей итерации
            df2['fscore' + str(i)] = importance
        except ValueError:
            # выдается ошибка, если задан алгоритм, в котором 
            # нет атрибута feature_importances_
            print("this clf doesn't have the feature_importances_ method.  Only Sklearn tree based methods allowed")
        
        # нормируем значения важности
        df2['fscore'+str(i)] = df2['fscore'+str(i)] / df2['fscore'+str(i)].sum()
        # объединяем датафреймы df и df2, т.е. к df добавляется столбец со 
        # значениями важности, найденными в текущей i-ой итерации
        df = pd.merge(df, df2, on='feature', how='outer')
        # если не задан "тихий" режим, печатаем информацию
        # о текущем раунде и итерации
        if not silent:
            print("Round: ", this_round, " iteration: ", i)

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
def _BoostARoota(x, y, metric, clf, cutoff, iters, max_rounds, delta, silent):
    """
    Функция проходит цикл, ожидая изменения критерия остановки
    :параметр x: массив признаков X (если есть категориальные переменные,
        нужно выполнить дамми-кодирование)
    :параметр y: массив меток зависимой переменной
    :параметр metric: оптимизируемая метрика
    :возвращает: имена переменных, которые нужно сохранить
    """
    # создаем копию обучающего массива признаков
    new_x = x.copy()
    
    # выполняем цикл до тех пор, пока переменная crit не изменится
    # выставляем в ноль счетчик раундов
    i = 0
    while True: # внутри этого цикла мы уменьшаем набор данных на каждом раунде
        # увеличиваем счетчик раундов
        i += 1
        
        # если параметр clf задан по умолчанию, то для отбора
        # признаков используется алгоритм XGBoost
        if clf is None:
            # вызывается функция _reduce_vars_xgb, которая возвращает критерий 
            # остановки и уменьшенный на данном раунде список предикторов
            crit, keep_vars = _reduce_vars_xgb(new_x,
                                               y,
                                               metric=metric,
                                               this_round=i,
                                               cutoff=cutoff,
                                               n_iterations=iters,
                                               delta=delta,
                                               silent=silent)
            
        # в противном случае используется алгоритм из библиотеки 
        # sklearn, в котором есть атрибут feature_importances_
        else:
            # вызывается функция _reduce_vars_sklearn (алгоритм передается
            # через параметр clf), которая возвращает критерий остановки
            # и уменьшенный на данном раунде список предикторов
            crit, keep_vars = _reduce_vars_sklearn(new_x,
                                                   y,
                                                   clf=clf,
                                                   this_round=i,
                                                   cutoff=cutoff,
                                                   n_iterations=iters,
                                                   delta=delta,
                                                   silent=silent)
            
        # если критерий остановки принял значение True 
        # или достигнуто максимальное количество раундов
        if crit | (i >= max_rounds):
            break
            # то выйти из цикла и использовать keep_vars в качестве
            # итогового списка отобранных переменных
        # в противном случае
        else:
            # создаем копию массива из списка признаков keep_vars
            new_x = new_x[keep_vars].copy()
    # если режим не является "тихим"
    if not silent:
        # напечатать сообщение об успешном завершении работы алгоритма
        print("BoostARoota ran successfully! Algorithm went through ", i, " rounds.")
    return keep_vars

