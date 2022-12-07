import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
import lightgbm as lgb
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, base_pipe, StratifiedGroupKFold, AutoTrees, AutoSelection
# Библиотека для подбора гиперпараметров
import optuna
from optuna.samplers import TPESampler
# импортируем функции для визуализации
from optuna.visualization import plot_slice, plot_contour, plot_optimization_history

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Поиск оптимального набора гиперпараметров", page_icon="🎛")

st.markdown("# Поиск оптимального набора гиперпараметров")
st.markdown("### 👈 Необходимо задать минимальный и максимальный порог для параметров")

st.sidebar.header("Optuna параметры")

useful_columns = pickle.load(open('./output/columns_after_deep_selection.sav','rb'))
target = useful_columns['target']

@st.cache
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

@st.cache
def user_params(need_columns):

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения (random_state)', 1,100,42)

    trials = st.sidebar.slider('Количество попыток подбора параметров (n_trials))', 5,30,15)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    with st.sidebar.expander('Шаг обучение (learning_rate)'):
        learning_rate_min = st.sidebar.slider("Минимальный (learning_rate)",0.01,1.0)
        learning_rate_max = st.sidebar.slider("Максимальный (learning_rate)",0.01,1.0,1.0)

    with st.sidebar.expander('Максимальная глубина (max_depth)'):
        max_depth_min = st.sidebar.slider("Минимальный (max_depth)",1.0,9.0)
        max_depth_max = st.sidebar.slider("Максимальный (max_depth)",1.0,15.0,9.0)

    with st.sidebar.expander('Регуляризация (reg_alpha)'):
        reg_alpha_min = st.sidebar.slider("Минимальный (reg_alpha)",0.01,1.0)
        reg_alpha_max = st.sidebar.slider("Максимальный (reg_alpha)",0.01,1.0,1.0)

    with st.sidebar.expander('Регуляризация (reg_lambda)'):
        reg_lambda_min = st.sidebar.slider("Минимальный (reg_lambda)",0.01,1.0)
        reg_lambda_max = st.sidebar.slider("Максимальный (reg_lambda)",0.01,1.0,1.0)

    optuna_params = {'learning_rate_min':learning_rate_min,
    'learning_rate_max':learning_rate_max,
    'max_depth_min':max_depth_min,
    'max_depth_max':max_depth_max,
    'reg_alpha_min':reg_alpha_min,
    'reg_alpha_max':reg_alpha_max,
    'reg_lambda_min':reg_lambda_min,
    'reg_lambda_max':reg_lambda_max,
    'trials':trials}

    type_selection = ['forward','deep backward','выбрать']
    ind = type_selection.index('выбрать')
    col_selected = st.sidebar.selectbox("Тип задачи", type_selection, index=ind)

    return train_test_params, col_selected, optuna_params

train_test_params, col_selected, optuna_params = user_params(useful_columns)

strat = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=train_test_params['random_state'])

if col_selected == 'forward':
    features = useful_columns['f_features']
    num_columns = useful_columns['f_num_columns']
    cat_columns = useful_columns['f_cat_columns']
elif col_selected == 'deep backward':
    features = useful_columns['b_features']
    num_columns = useful_columns['b_num_columns']
    cat_columns = useful_columns['b_cat_columns']
else:
    st.warning('Выберите метод отбора, чтобы подгрузить списки полученных факторов!')
    st.stop()

st.write('Количество факторов: ', len(features))
st.write('Отобранные факторы: ', features)

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

    st.markdown("#### Начало поиска оптимального набора гиперпараметров!")
    st.write("---")

    def create_model(trial):

        param = {'learning_rate':trial.suggest_loguniform(name='learning_rate', low=optuna_params['learning_rate_min'], high=optuna_params['learning_rate_max']),
                 'n_estimators':1000,
                 'random_state':train_test_params['random_state'],
                 'n_jobs':-1,
                 'max_depth': trial.suggest_int(name="max_depth", low=optuna_params['max_depth_min'], high=optuna_params['max_depth_max']),
                 'subsample':trial.suggest_loguniform("subsample", 0.4, 1.0),
                 'colsample_bytree':trial.suggest_loguniform(name="colsample_bytree", low=0.4, high=1.0),
                 'reg_alpha': trial.suggest_loguniform(name='lambda_l1', low=optuna_params['reg_alpha_min'], high=optuna_params['reg_alpha_max']),
                 'reg_lambda': trial.suggest_loguniform(name='reg_lambda', low=optuna_params['reg_lambda_min'], high=optuna_params['reg_lambda_max'])
        }

        fit_params = {
            'early_stopping_rounds':100,
            'eval_metric':['logloss', 'auc'],
            'verbose':False}

        # создаем экземпляр класса XGBClassifier
        lgb_model = lgb.LGBMClassifier(**param)

        model = AutoTrees(
            main_estimator = lgb_model,
            main_fit_params = fit_params,
            main_prep_pipe = prep_pipe,
            main_features = num_columns+cat_columns,

            X_train=X_train,
            y_train=y_train,
            main_metric='roc_auc',
            model_type = 'lightboost')

        return model

    def objective(trial):
        my_model = create_model(trial)
        my_model.model_fit_cv(strat=strat)
        return my_model.get_mean_cv_scores()

    sampler = TPESampler(seed=train_test_params['random_state'])

    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=optuna_params['trials'])

    best_params = study.best_params
    st.write(f'Параметры после подбора: {best_params}')
    st.write("---")


    st.markdown("#### График оптимизации подбора параметров")
    st.write(plot_optimization_history(study))
    st.write("---")


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

    params.update(best_params)

    columns_after_optuna = {
                            'features':features,
                            'num_columns':num_columns,
                            'cat_columns':cat_columns,
                            'target':target}

    pickle.dump(columns_after_optuna,open('./output/columns_after_optuna.sav', 'wb'))
    pickle.dump(params,open('./output/params_optuna.sav', 'wb'))

    st.write('### Поиск оптимального набора гиперпараметров завершен! ✅')

st.button("Перезапуск")
