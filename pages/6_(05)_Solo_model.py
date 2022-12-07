import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
import lightgbm as lgb
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, base_pipe, BalanceCover
from autobinary.utils.folders import create_folder

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Финализация модели", page_icon="✅")

st.markdown("# Финализация модели")
st.markdown("### 👈 Необходимо задать параметры для отрисовки BalanceCover мертрики")

st.sidebar.header("Optuna параметры")

useful_columns = pickle.load(open('./output/columns_after_optuna.sav','rb'))
features = useful_columns['features']
num_columns = useful_columns['num_columns']
cat_columns = useful_columns['cat_columns']
target = useful_columns['target']

final_params = pickle.load(open('./output/params_optuna.sav','rb'))

def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

def user_params(need_columns):

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения (random_state)', 1,100,42)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    return train_test_params

train_test_params = user_params(useful_columns)

st.write('Количество факторов: ', len(features))
st.write('Отобранные факторы: ', features)

link_5 = st.text_input('Введите ссылку на датасет')
if link_5 == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link_5)
    st.write('Датасет загружен')

#if st.button('Старт обучения'):
#    st.markdown('# Процесс запущен!')

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

    prep_pipe.fit(X_train[features], y_train)

    X_train_final = prep_pipe.transform(X_train[features])
    X_valid_final = prep_pipe.transform(X_valid[features])

    model_final = lgb.LGBMClassifier(**final_params)
    model_final.fit(X_train_final, y_train)

    # Кастомная метрика

    metric_train = pd.DataFrame()
    metric_train['target'] = y_train
    metric_train['proba'] = model_final.predict_proba(X_train_final)[:, 1]
    metric_train = metric_train.sort_values('proba', ascending=False).reset_index(drop=True)

    metric_valid = pd.DataFrame()
    metric_valid['target'] = y_valid
    metric_valid['proba'] = model_final.predict_proba(X_valid_final)[:, 1]
    metric_valid = metric_valid.sort_values('proba', ascending=False).reset_index(drop=True)

    metr_train = BalanceCover(metric_train, target='target')

    st.write("Размер обучающего множества:", X_train_final.shape)
    st.write("Количество таргета на обучении:", y_train.sum())

    st.markdown("#### BalanceCover для обучающего множества")
    bin_min_tr = st.text_input("Шаг бина (трейн)")
    bin_max_tr = st.text_input("Максимальное количество наблюдений (трейн)")

    if bin_min_tr=='' or bin_max_tr=='':
        st.warning('Задать параметры BalanceCover')
        st.stop()
    else:
        metr_train.calc_scores(int(bin_min_tr), int(bin_max_tr))

    st.write('Визуализация метрики на трейне')
    metr_train.plot_scores()
    st.pyplot(bbox_inches='tight')
    st.write('---')
    st.write('Табличное представление')
    st.write(metr_train.output)


    metr_valid = BalanceCover(metric_valid, target='target')

    st.write("Размер тестового множества:", X_valid_final.shape)
    st.write("Количество таргета на тесте:", y_valid.sum())

    st.markdown("#### BalanceCover для тестового множества")
    bin_min_te = st.text_input("Шаг бина (тест)")
    bin_max_te = st.text_input("Максимальное количество наблюдений (тест)")

    if bin_min_te=='' or bin_max_te=='':
        st.warning('Задать параметры BalanceCover')
        st.stop()
    else:
        metr_valid.calc_scores(int(bin_min_te), int(bin_max_te))

    st.write('Визуализация метрики на тесте')
    metr_valid.plot_scores()
    st.pyplot(bbox_inches='tight')
    st.write('---')
    st.write('Табличное представление')
    st.write(metr_train.output)

    path = 'final_results'

    create_folder(path)
    pickle.dump(prep_pipe,open('./{}/prep_pipe_final.sav'.format(path), 'wb'))
    pickle.dump(num_columns,open('./{}/num_columns_final.sav'.format(path), 'wb'))
    pickle.dump(cat_columns,open('./{}/cat_columns_final.sav'.format(path), 'wb'))
    pickle.dump(model_final,open('./{}/model_final.sav'.format(path), 'wb'))

    st.write('### Финальный набор атрибутов получен! ✅')

st.button("Перезапуск")
