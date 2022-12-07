import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, PermutationSelection, base_pipe
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Обработка и первчиный отбор факторов", page_icon="📊")

st.markdown("# Обработка и первичный отбор факторов")
st.markdown("""### 👈 Необходимо задать параметры для отбора факторов
1) Числовые переменные;
2) Категориальные переменные;
3) Целевая переменная
""")

st.sidebar.header("Обработка и первичный отбор факторов")

useful_columns = pickle.load(open('./output/columns_after_eda.sav','rb'))
useful_columns.append('выбрать')

def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

def user_params(need_columns):

    num_columns = st.sidebar.multiselect("Числовые переменные", need_columns)
    cat_columns = st.sidebar.multiselect("Категориальные переменные", need_columns)
    ind = need_columns.index('выбрать')
    target = st.sidebar.selectbox("Целевая переменная", need_columns, index=ind)

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения', 1,100,42)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    model_permutation = ['xgboost','catboost','lightboost','decisiontree', 'randomforest','выбрать']
    ind = model_permutation.index('выбрать')
    model = st.sidebar.selectbox("Модель для отбора признаков", model_permutation, index=ind)
    task = ['classification','regression','multiclassification','выбрать']
    ind = task.index('выбрать')
    task_type = st.sidebar.selectbox("Тип задачи", task, index=ind)
    depth = st.sidebar.slider('Глубина для анализа значимости факторов', 1,10,3)
    n_iter = st.sidebar.slider('Количество перемешиваний в Permutation Importnce', 5,30,10)

    selection_params = {
                        'model':model,
                        'task_type':task_type,
                        'depth':depth,
                        'n_iter':n_iter
                        }

    return num_columns, cat_columns, target, train_test_params, selection_params

num_columns, cat_columns, target, train_test_params, selection_params = user_params(useful_columns)
st.write("Количество числовых признаков: ", num_columns, len(num_columns))
st.write("Количество категориальных признаков: ", cat_columns, len(cat_columns))
st.write('Целевая переменная:', target)

if target=='выбрать':

    st.stop()
st.success("Все переменные обозначены")

link_1 = st.text_input('Введите ссылку на датасет')
if link_1 == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link_1)
    st.write('Датасет загружен')

if st.button('Старт обучения'):
    st.markdown('# Процесс запущен!')

    st.write('### Разбиваем на обучающее и тестовое множества в отношении:',
    1-train_test_params['test_size'],'-',train_test_params['test_size'])

    X_train, X_valid, y_train, y_valid = train_test_split(
        sample,
        sample[target],
        test_size=train_test_params['test_size'],
        stratify=sample[target],
        random_state=train_test_params['random_state']
    )

    st.write(" * Размер обучающего множества: ", len(X_train))
    st.write(" * Размер тестового множества: ", len(X_valid))
    st.write("---")
    # Разделение тренировочной выборки на подмножества для расчета Permutation Importance

    folds_perm = list(StratifiedKFold(n_splits=4,
                                      shuffle=True,
                                      random_state=42).split(X_train,y_train))

    df_train_perm = X_train.iloc[folds_perm[3][0]]
    print('Train permutation: ', df_train_perm.shape, ';','Target rate: ',  df_train_perm[target].mean())

    df_test_perm = X_train.iloc[folds_perm[3][1]]
    print('Test permutation: ', df_test_perm.shape, ';','Target rate: ', df_test_perm[target].mean())

    # Трансформация категориальных и числовых признаков для корреляционного анализа и Permutation Importance
    features = num_columns + cat_columns

    if len(num_columns)>0 and len(cat_columns)>0:
        prep_pipe = base_pipe(
            num_columns=num_columns,
            cat_columns=cat_columns,
            kind='all')

    elif len(num_columns)==0 and len(cat_columns)>0:
        prep_pipe = base_pipe(
            cat_columns=cat_columns,
            kind='cat')

    elif len(num_columns)>0 and len(cat_columns)==0:
        prep_pipe = base_pipe(
            num_columns=num_columns,
            kind='num')

    prep_pipe.fit(df_train_perm[features], df_train_perm[target])

    X_train_perm = prep_pipe.transform(df_train_perm[features])
    y_train_perm = df_train_perm[target]

    X_test_perm = prep_pipe.transform(df_test_perm[features])
    y_test_perm = df_test_perm[target]

    if selection_params['task_type']=='classification':
        if selection_params['model']=='xgboost':
            params_m = {'eta':0.01,
                      'n_estimators':500,
                      'subsample':0.9,
                      'max_depth':6,
                      'objective':'binary:logistic',
                      'n_jobs':-1,
                      'random_state':train_test_params['random_state'],
                      'eval_metric':'logloss'}
        elif selection_params['model']=='catboost':
            params_m = {'learning_rate':0.01,
                      'iterations':500,
                      'subsample':0.9,
                      'depth':6,
                      'loss_function':'Logloss',
                      'thread_count':-1,
                      'random_state':train_test_params['random_state'],
                      'verbose':0}
        elif selection_params['model']=='lightboost':
            params_m = {'learning_rate':0.01,
                      'n_estimators':500,
                      'subsample':0.9,
                      'max_depth':6,
                      'objective':'binary',
                      'metric':'binary_logloss',
                      'n_jobs':-1,
                      'random_state':train_test_params['random_state'],
                      'verbose':-1}
        elif selection_params['model']=='decisiontree':
            params_m = {'criterion':'gini',
                     'max_depth':6,
                     'random_state':train_test_params['random_state']}
        elif selection_params['model']=='randomforest':
            params_m = {'criterion':'gini',
                     'max_depth':6,
                     'random_state':train_test_params['random_state'],
                     'n_estimators':500}

    st.write(" * Количество числовых признаков было: ", len(num_columns))
    st.write(" * Количество категориальных признаков было: ", len(cat_columns))
    st.write("---")

    st.markdown("### Инициализируется класс для первичного отбора факторов")

    # Инициализация класса Permutation Importance

    perm_imp = PermutationSelection(
        model_type=selection_params['model'],
        model_params=params_m,
        task_type=selection_params['task_type'])

    # Анализ значимости признаков по глубине обучения алгоритма (глубина меняется от 1 до max_depth). Отбираются только те, у кого средняя значимость > 0.

    fi, fi_rank, depth_features, rank_features = perm_imp.depth_analysis(
        X_train=X_train_perm,
        y_train=y_train_perm,
        features=features,
        max_depth=selection_params['depth'])

    # Обновляем факторы

    features = list(filter(lambda x: x in features, depth_features))
    num_columns = list(filter(lambda x: x in features, num_columns))
    cat_columns = list(filter(lambda x: x in features, cat_columns))

    st.markdown("##### 1) Анализ относительно глубины алгоритма проведен")
    st.dataframe(fi)
    st.write("---")
    # Обновляем трансформацию факторов

    if len(num_columns)>0 and len(cat_columns)>0:
        prep_pipe = base_pipe(
            num_columns=num_columns,
            cat_columns=cat_columns,
            kind='all')

    elif len(num_columns)==0 and len(cat_columns)>0:
        prep_pipe = base_pipe(
            cat_columns=cat_columns,
            kind='cat')

    elif len(num_columns)>0 and len(cat_columns)==0:
        prep_pipe = base_pipe(
            num_columns=num_columns,
            kind='num')

    prep_pipe.fit(df_train_perm[features], df_train_perm[target])

    X_train_perm = prep_pipe.transform(df_train_perm[features])
    X_test_perm = prep_pipe.transform(df_test_perm[features])

    # Обучаем Permutation Importance

    perm_imp.fit(
        X_train=X_train_perm,
        y_train=y_train_perm)

    if selection_params['task_type'] == 'classification':
        metric = roc_auc_score
        higher_is_better = True
    elif selection_params['task_type'] == 'regression':
        metric = mean_squared_error
        higher_is_better = False
    elif selection_params['task_type'] == 'multiclassification':
        metric = average_precision_score
        higher_is_better = True

    # Рассчитываем метрики для каждого фактора

    perm_table = perm_imp.calculate_permutation(
        X_test=X_test_perm,
        y_test=y_test_perm,
        n_iter=selection_params['n_iter'],
        permute_type='kib',
        n_jobs=-1,
       metric=metric,
       higher_is_better=higher_is_better
    )

    st.markdown("##### 2) Permutation Importance проведен")
    st.write("---")

    # Отрисовка топ факторов по Permutation Importance
    st.markdown("### Важность факторов после Permutation Importance (топ 10)")
    perm_imp.permutation_plot(figsize=(16,12), top=10)
    st.pyplot(bbox_inches='tight')

    # Обновляем признаки

    features = perm_imp.select_features()
    num_columns = list(filter(lambda x: x in features, num_columns))
    cat_columns = list(filter(lambda x: x in features, cat_columns))

    st.write(" * Количество числовых признаков стало: ", len(num_columns))
    st.write(" * Количество категориальных признаков стало: ", len(cat_columns))
    st.write('---')

    columns_after_permutation = {
                                'features':features,
                                'num_columns':num_columns,
                                'cat_columns':cat_columns,
                                'target':target}

    pickle.dump(columns_after_permutation,open('./output/columns_after_permutation.sav', 'wb'))
    st.write('##### Факторы после обработки и первичного отбора сохранены! ✅')

st.button("Перезапуск")
