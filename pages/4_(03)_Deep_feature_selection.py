import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
import lightgbm as lgb
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, base_pipe, StratifiedGroupKFold, AutoTrees, AutoSelection
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Глубокий отбор факторов с помощью Forward Selection и Backward Selection", page_icon="⚙️")

st.markdown("# Последовательный отбор факторов по кросс-валидации")
st.markdown("### 👈 Необходимо задать параметры для анализа")

st.sidebar.header("Forward / Backward Selection")

useful_columns = pickle.load(open('./output/columns_after_permutation.sav','rb'))
features = useful_columns['features']
num_columns = useful_columns['num_columns']
cat_columns = useful_columns['cat_columns']
target = useful_columns['target']

@st.cache
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

def user_params(need_columns):

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения', 1,100,42)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    tol = st.sidebar.slider('Порог разницы в значении метрики при Backward Selection', 0.001,0.1,0.01)

    return train_test_params, tol

train_test_params, tol = user_params(useful_columns)

link = st.text_input('Введите ссылку на датасет')
if link == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link)
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

    st.markdown("## Начало глубокого отбора факторов!")
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

    selection = AutoSelection(base_pipe=base_pipe,
                              num_columns=num_columns,
                              cat_columns=cat_columns,
                              main_fit_params=fit_params,
                              main_estimator=lgb_model,

                              X_train=X_train,
                              y_train=y_train,
                              main_metric='roc_auc',
                              model_type='lightboost')

    fselection_res = selection.forward_selection(strat=strat)

    st.markdown("##### Прямой последовательный отбор факторов Forward Selection")
    selection.plot_forward(figsize=(16,8))
    st.pyplot(bbox_inches='tight')
    st.write("---")

    st.markdown("##### Обратный отбор факторов Deep Backward Selection")
    deep_bselection_res = selection.deep_backward_selection(strat=strat,tol=tol)
    st.write("---")

    st.markdown("### Результаты отбора факторов и проверка на тестовом отложенном множесте")
    st.dataframe(selection.report(X_valid,y_valid))
    st.write("---")

    f_features = fselection_res['features_stack']
    f_num_columns = list(filter(lambda x: x in f_features, num_columns))
    f_cat_columns = list(filter(lambda x: x in f_features, cat_columns))

    b_features = deep_bselection_res['features_stack']
    b_num_columns = list(filter(lambda x: x in b_features, num_columns))
    b_cat_columns = list(filter(lambda x: x in b_features, cat_columns))

    columns_after_deep_selection = {
                                'f_features':f_features,
                                'f_num_columns':f_num_columns,
                                'f_cat_columns':f_cat_columns,
                                'b_features':b_features,
                                'b_num_columns':b_num_columns,
                                'b_cat_columns':b_cat_columns,
                                'target':target}

    pickle.dump(columns_after_deep_selection,open('./output/columns_after_deep_selection.sav', 'wb'))

    st.write('### Глубокий отбор факторов завершен, все результаты сохранены! ✅')

st.button("Перезапуск")
