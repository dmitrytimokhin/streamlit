import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Наша разработанная библиотека
from autobinary import TargetPlot
from autobinary.utils.folders import create_folder
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Загрузка данных и EDA", page_icon="📂")

st.markdown("# Загрузка данных и Exploratory Data Analysis (EDA)")
st.markdown("### 👈 Необходимо задать параметры для анализа данных")
st.sidebar.header("Загрузка данных и EDA")

@st.cache
def user_params():
    types = ['bin','reg','выбрать']
    default_ind = types.index('выбрать')
    type_target = st.sidebar.selectbox("Тип таргета", types, default_ind)
    bins = st.sidebar.slider('Число бинов',1,30,10)
    left_quant = st.sidebar.slider('Левое отсечение', 0.00,1.00,0.00)
    right_quant = st.sidebar.slider('Правое отсечение', 0.00,1.00,0.00)
    types = [True,False,'выбрать']
    default_ind = types.index('выбрать')
    log = st.sidebar.selectbox("Логарифмирование таргета", types, default_ind)

    params = {
        'type_target':type_target,
        'bins':bins,
        'left_quant':left_quant,
        'right_quant':right_quant,
        'spec_value':False,
        'log':log}

    return params

params_eda = user_params()

@st.cache
def load_dataset(data_link):
    dataset = pd.read_csv(data_link)
    return dataset

use_columns = {'use':None,
                'target':None}

link = st.text_input('Введите ссылку на датасет')
if link == '':
    st.write('Датасет не загружен')
    st.stop()
else:
    sample = load_dataset(link)
    st.write('Датасет загружен')

##if st.button('Старт обучения'):
##    st.markdown('# Процесс запущен!')

    st.dataframe(sample.head())

    st.markdown('### Статистика по данным')
    st.write('Размерность данных: ', sample.shape)

    check_nans = pd.DataFrame(sample.isna().sum()).rename(columns={0:'Nans'}).sort_values(by='Nans',ascending=False)
    check_nans['Nans_percent'] = (check_nans.Nans/len(sample))*100

    drop_columns = list(check_nans[check_nans.Nans_percent>95].index)
    st.write('Количество признаков с пропусками >95%: ', len(drop_columns))
    st.dataframe(check_nans[check_nans.Nans_percent>95])

    st.markdown('## Визуализация числового фактора относительно целевой переменной')
    all_columns = sample.columns.tolist() + ['выбрать']
    default_ind = all_columns.index('выбрать')
    use_columns['target'] = st.selectbox("Целевая переменная", all_columns, index=default_ind)

    cnt_columns = sample.select_dtypes(include=[int,float]).columns.tolist()+['выбрать']
    default_ind = cnt_columns.index('выбрать')
    use_columns['use'] = st.selectbox("Числовой переменная", cnt_columns, index=default_ind)

    if use_columns['use']=='выбрать' or use_columns['target']=='выбрать':
        st.stop()
        st.warning('Пожалуйста выберите целевую и числовую переменную')
    st.success(f"Выбранная числовая переменная: {use_columns['use']!r}")

    tt = TargetPlot(sample=sample,
                    feature=use_columns['use'],
                    target=use_columns['target'],
                    type_target=params_eda['type_target'],
                    bins=params_eda['bins'],
                    left_quant=params_eda['left_quant'],
                    right_quant=params_eda['right_quant'],
                    spec_value=params_eda['spec_value'],
                    log=params_eda['log']
                    )

    st.markdown('### Визуализация результатов')
    st.dataframe(tt.get_bin_table())

    st.write("""
    #### Кривая полученной калибровки
    """)
    tt.get_target_plot()
    st.pyplot(bbox_inches='tight')

    useful_columns = list(set(sample.columns)-set(drop_columns))

    path = 'output'
    create_folder(path)
    pickle.dump(useful_columns,open('./{}/columns_after_eda.sav'.format(path), 'wb'))

    st.write('#### Факторы после анализа пропущенных значений сохранены')

st.button("Перезапуск")
