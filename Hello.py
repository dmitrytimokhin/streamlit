import streamlit as st

st.set_page_config(
    page_title="Привет, пользователь",
    page_icon="👋",
)

st.write("# Добро пожаловать к нам на платформу!")

st.sidebar.success("Выберите этап обучения")

st.markdown(
    """
    На данном сервере мы постараемся продемонстрировать вам автоматическую работу
    нашей библиотеки [AutoBinary](https://github.com/Vasily-Sizov/autobinary_framework)

    **👈 Выберите этап обучения**
    ### Последовательность обучения:
    - Загрузка данных и EDA;
    - Обработка и первичный отбор факторов;
    - Обучение по кросс-валидационной схеме;
    - Глубокий отбор факторов;
    - Подбор гиперпараметров алгоритма;
    - Финализация модели;
    - Интерпретация результатов.

    ### Пример полного обучения модели можно посмотреть в открытом доступе:
    - [Посмотреть пример на данных о крушении Титаника](https://github.com/Vasily-Sizov/autobinary_framework/blob/main/%5B2022.09.19%5D%20Example%20of%20Full%20fitting%20model.ipynb)
    - Ссылка на данные крушения Титаника: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""
)

import stramlit
import autobinary
st.write()
st.write('streamlit version:',streamlit.__version__)
st.write('autobinary version:', autobinary.__version__)
