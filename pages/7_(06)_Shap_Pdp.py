import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score
import lightgbm as lgb
# Наша разработанная библиотека
# Наша разработанная библиотека
from autobinary import SentColumns, CatBoostEncoder, PermutationSelection, base_pipe, PlotShap, PlotPDP

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Визуализация и интерпретация результатов", page_icon="🎉")

st.markdown("# Визуализация и интерпретация результатов")
st.markdown("### 👈 Необходимо задать параметры для визуализации")

st.sidebar.header("Параметры визуализации")

useful_columns = pickle.load(open('./output/columns_after_optuna.sav','rb'))
features = useful_columns['features']
num_columns = useful_columns['num_columns']
cat_columns = useful_columns['cat_columns']
target = useful_columns['target']

prep_pipe = pickle.load(open('./final_results/prep_pipe_final.sav','rb'))

model = pickle.load(open('./final_results/model_final.sav','rb'))

def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

def user_params(features):

    test_size = st.sidebar.slider('Отношение разбиения трейн-тест', 0.1,1.0,0.3)
    random_state = st.sidebar.slider('Фактор фиксации решения (random_state)', 1,100,42)

    train_test_params = {'test_size':test_size,
                        'random_state':random_state}

    top_feat = st.sidebar.slider('Число факторов для отрисовки', 1,len(features),1)

    percent_sample = st.sidebar.slider('Процент подвыборки для отрисовки фактора PDP', 0.15,1.0,0.3)

    type_selection = [True,False,'выбрать']
    ind = type_selection.index('выбрать')
    save_report = st.sidebar.selectbox("Сохранить отчеты", type_selection, index=ind)

    report_params = {'top_feat':top_feat,
                     'percent_sample':percent_sample,
                     'save_report':save_report}

    return train_test_params, report_params

train_test_params, report_params = user_params(features)

st.write('Количество факторов: ', len(features))
st.write('Отобранные факторы: ', features)

link_6 = st.text_input('Введите ссылку на датасет')
if link_6 == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link_6)
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
    st.write("---")

    X_train = prep_pipe.transform(X_train[features])

    # задаем класс
    c_shap = PlotShap(model=model, sample=X_train)
    # обучаем
    c_shap.fit_shap()

    st.markdown("#### Shap Importance для обучающего множества")
    c_shap.create_plot_shap(plot_type='bar', number_features=report_params['top_feat'])
    st.pyplot(bbox_inches='tight')
    st.write('---')

    if report_params['save_report']:
        c_shap.create_shap_report(path_base='./final_results/shap_report')
        st.write('Отчет Shap сохранен!')
    else:
        st.write('Отчет не сохранен!')

    feats = pd.DataFrame({'imp':model.feature_importances_,'feat':features}).sort_values('imp', ascending=False).feat.tolist()
    pdp_num_columns = list(filter(lambda x: x in num_columns, feats))
    st.markdown("#### PDP plot для обучающего множества")
    pdp_plot = PlotPDP(model=model,X=X_train,main_features=pdp_num_columns)
    pdp_plot.create_feature_plot(save=report_params['save_report'], frac_to_plot=report_params['percent_sample'], path='./final_results/pdp_report')
    st.pyplot(bbox_inches='tight')
    st.write('---')

    if report_params['save_report']:
        pdp_plot.create_pdp_report(path_base='./final_results/pdp_report')
        st.write('Отчет PDP сохранен!')
    else:
        st.write('Отчет не сохранен!')

    st.write('### Процесс обучения и интерпретации результатов завершен! ✅')

st.button("Перезапуск")
