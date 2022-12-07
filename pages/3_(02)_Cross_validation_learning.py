import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
import lightgbm as lgb
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, base_pipe, StratifiedGroupKFold, AutoTrees
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Кросс-валидационное обучение и контроль качества", page_icon="⏳")

st.markdown("# Кросс-валидационное обучение")
st.markdown("### 👈 Необходимо задать параметры для обучения")

st.sidebar.header("Кросс-валидационное обучение")

useful_columns = pickle.load(open('./output/columns_after_permutation.sav','rb'))
features = useful_columns['features']
num_columns = useful_columns['num_columns']
cat_columns = useful_columns['cat_columns']
target = useful_columns['target']

def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

def user_params(need_columns):

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения', 1,100,42)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    return train_test_params

train_test_params = user_params(useful_columns)

link_2 = st.text_input('Введите ссылку на датасет')
if link_2 == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link_2)
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

    params = {
        'learning_rate':0.01,
        'n_estimators':1000,
        'subsample':0.9,
        'colsample_bytree':0.6,
        'max_depth':6,
        'objective':'binary',
        'n_jobs':-1,
        'random_state':train_test_params['random_state']
    }

    fit_params = {
        'early_stopping_rounds':100,
        'eval_metric':['logloss', 'auc'],
        'verbose':25}

    # создаем экземпляр класса LightGBM
    lgb_model = lgb.LGBMClassifier(**params)

    # задаем стратегию проверки
    strat = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42)

    model = AutoTrees(
        main_estimator=lgb_model,
        main_fit_params=fit_params,
        main_prep_pipe = prep_pipe,
        main_features=features,

        X_train=X_train,
        y_train=y_train,
        main_metric='roc_auc',
        model_type = 'lightboost')

    model.model_fit_cv(strat=strat)

    st.write("### Средняя метрика на кросс - валидации = ", model.get_mean_cv_scores())
    st.write("---")
    st.write("Контроль значения метрик на фолдах")
    st.dataframe(model.get_extra_scores())
    st.write("---")
    st.write("Значения средней значимости факторов по кросс - валидации")
    st.dataframe(model.get_fi())
    st.write("---")

    st.markdown("##### Значение ROC - AUC на тестовых фолдах")
    print(model.get_rocauc_plots())
    st.pyplot(bbox_inches='tight')
    st.write("---")

    st.write('### Обучение и визуализация кросс - валидации выполнена! ✅')

st.button("Перезапуск")
