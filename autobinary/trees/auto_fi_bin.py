import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ..utils.html_style import settings_style, start_report, end_report
from ..utils.to_html import get_img_tag, df_html
from ..utils.folders import create_folder

class XGBFeatureImportanceBin:
    """
    """
    def __init__(self, X_train, y_train, cv, pipeline, features, model):
        """
        """
        self.X_train = X_train.reset_index(drop=True).copy()
        self.y_train = y_train.reset_index(drop=True).copy()
        self.cv = cv
        self.pipeline = pipeline
        self.features = features
        self.model = model
        
    def calculate_importance(self, groups=None):
        """
        """
        i = 0
        self.results_ = {}
        self.fi_ = []
        self.scores_dict = {}
        self._test_group = []

        # Пробегаемся по фолдам
        for train_ix, test_ix in tqdm(self.cv.split(self.X_train[self.features], self.y_train, groups=groups)):
            i+=1
            
            if groups is not None:
                groups_train, groups_test = groups.loc[train_ix], groups.loc[test_ix]
                self._test_group.append(len(set(groups_train).intersection(set(groups_test))))


            X_train_f, X_test_f = self.X_train[self.features].loc[train_ix], self.X_train[self.features].loc[test_ix]
            y_train_f, y_test_f = self.y_train.loc[train_ix], self.y_train.loc[test_ix]

            # Применяем конвейер
            self.pipeline.fit(X_train_f, y_train_f)
            
            if isinstance(self.pipeline.transform(X_train_f), pd.DataFrame):
                X_train_new = self.pipeline.transform(X_train_f)
                X_test_new = self.pipeline.transform(X_test_f)
            else:
                X_train_new = pd.DataFrame(self.pipeline.transform(X_train_f))
                X_train_new.columns = X_train_f.columns
                X_test_new = pd.DataFrame(self.pipeline.transform(X_test_f))
                X_test_new.columns = X_test_f.columns
    
            print(f'Обучение {i} - ой модели:')
            
            # Обучаем модель
            eval_set1 = [(X_train_new, y_train_f), (X_test_new, y_test_f)]
            
            # строим модель
            self.model.fit(
                X_train_new, 
                y_train_f, 
                early_stopping_rounds=50,
                eval_metric=['logloss', 'aucpr', 'auc'], 
                eval_set=eval_set1, 
                verbose=50)
    
            # оцениваем дискриминирующую способность модели xgboost
            roc_train = roc_auc_score(y_train_f, self.model.predict_proba(X_train_new)[:, 1])
            roc_valid = roc_auc_score(y_test_f, self.model.predict_proba(X_test_new)[:, 1])
            
            fpr_tr, tpr_tr, _ = roc_curve(y_train_f,  self.model.predict_proba(X_train_new)[:, 1])
            fpr_vl, tpr_vl, _ = roc_curve(y_test_f,  self.model.predict_proba(X_test_new)[:, 1])
            
            dict_temp = {'fpr_tr': fpr_tr, 'tpr_tr': tpr_tr, 'roc_train': roc_train, 
                         'fpr_vl': fpr_vl, 'tpr_vl': tpr_vl, 'roc_valid': roc_valid}
            
            self.scores_dict[f'scores_{i}'] = dict_temp
            
            print("AUC на обучающей выборке: {:.3f}".format(roc_train))
            print("AUC на проверочной выборке: {:.3f}".format(roc_valid))
            
            self.results_[f'eval_result_{i}'] = self.model.evals_result()
            
            self.fi_.append(self.model.get_booster().get_score(importance_type='gain'))
    
    def get_importances(self):
        """
        """
        df_fi = pd.DataFrame(self.fi_).T
        df_fi = df_fi.fillna(0)
        df_fi.columns = ['importance '+ str(idx) for idx in range(len(self.fi_))]

        # получаем усредненные важности признаков и выводим в порядке убывания
        df_fi['mean_importance'] = df_fi.mean(axis=1)
        df_fi = df_fi.sort_values('mean_importance', ascending=False)
        df_fi = df_fi.reset_index()
        
        return df_fi
        
    def curve_plot(self, metrics, path: str='-1', label = -1):
        """
        """
        # записываем значения метрик
        results = metrics.copy()
        # записываем количество итераций
        epochs = len(results['validation_0']['auc'])
        # задаем диапазон значений (итераций) для оси x
        x_axis = range(0, epochs)

        # строим график логистической функции потерь
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['logloss'], label='Обучающая выборка')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Проверочная выборка')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.xlabel('Iterations')
        plt.title('XGBoost Log Loss')
        plt.grid()
        plt.tight_layout()
        #plt.show()
        
        if path != '-1' and label != -1:
            plt.savefig(f'{path}/{label}_LogLoss_curve_plot.png')
            plt.close()

        # строим график AUC
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['auc'], label='Обучающая выборка')
        ax.plot(x_axis, results['validation_1']['auc'], label='Проверочная выборка')
        ax.legend()
        plt.ylabel('AUC')
        plt.xlabel('Iterations')
        plt.title('XGBoost AUC')
        plt.grid()
        plt.tight_layout()
        #plt.show()
        
        if path != '-1' and label != -1:
            plt.savefig(f'{path}/{label}_AUC_curve_plot.png')
            plt.close()
    
        # строим график AUCPR
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['aucpr'], label='Обучающая выборка')
        ax.plot(x_axis, results['validation_1']['aucpr'], label='Проверочная выборка')
        ax.legend()
        
        plt.ylabel('AUCPR')
        plt.xlabel('Iterations')
        plt.title('XGBoost AUCPR')
        plt.grid()
        plt.tight_layout()
        #plt.show()
    
        if path != '-1' and label != -1:
            plt.savefig(f'{path}/{label}_AUCPR_curve_plot.png')
            plt.close()
            
    def rocauc_plots(self, path: str='-1'):
        """
        """
        fig, ax = plt.subplots()
        for i in range(1, len(self.scores_dict)+1):
            plt.plot(self.scores_dict[f'scores_{i}']['fpr_tr'], 
                     self.scores_dict[f'scores_{i}']['tpr_tr'], 
                     label="fold {}, AUC={:.3f}".format(i, self.scores_dict[f'scores_{i}']['roc_train']))
    
        plt.plot([0,1], [0,1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve Analysis TRAIN', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.grid()
        plt.tight_layout()
        #plt.show()
            
        if path != '-1':
            plt.savefig(f'{path}/rocauc_plots_train.png')
            plt.close()
        
        fig, ax = plt.subplots()
        for i in range(1, len(self.scores_dict)+1):
            plt.plot(self.scores_dict[f'scores_{i}']['fpr_vl'], 
                     self.scores_dict[f'scores_{i}']['tpr_vl'], 
                     label="fold {}, AUC={:.3f}".format(i, self.scores_dict[f'scores_{i}']['roc_valid']))
    
        plt.plot([0,1], [0,1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curve Analysis VALIDATION', fontweight='bold', fontsize=15)
        plt.legend(prop={'size':13}, loc='lower right')
        plt.grid()
        plt.tight_layout()
        #plt.show()
            
        if path != '-1':
            plt.savefig(f'{path}/rocauc_plots_valid.png')
            plt.close()

    def _plots_learning_curve(self, results, path_base: str, name_fold):
    
        # записываем количество итераций
        epochs = len(results['validation_0']['logloss'])
        # задаем диапазон значений (итераций) для оси x
        x_axis = list(range(0, epochs))
    
        fig = make_subplots(
            rows=1, 
            cols=3, 
            subplot_titles=("XGBoost Log Loss", "XGBoost AUC", "XGBoost AUCPR"))

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_0']['logloss'],
                mode = "lines",
                name = "Обучающая выборка",
                marker = dict(color = 'rgba(0, 197, 255, 1)'),
                text= 'Обучающая выборка',
                legendgroup = '1'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_1']['logloss'],
                mode = "lines",
                name = "Проверочная выборка",
                marker = dict(color = 'rgba(255, 154, 0, 1)'),
                text= 'Проверочная выборка',
                legendgroup = '1'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_0']['auc'],
                mode = "lines",
                name = "Обучающая выборка",
                marker = dict(color = 'rgba(0, 197, 255, 1)'),
                text= 'Обучающая выборка',
                showlegend=False
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_1']['auc'],
                mode = "lines",
                name = "Проверочная выборка",
                marker = dict(color = 'rgba(255, 154, 0, 1)'),
                text= 'Проверочная выборка',
                showlegend=False
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_0']['aucpr'],
                mode = "lines",
                name = "Обучающая выборка",
                marker = dict(color = 'rgba(0, 197, 255, 1)'),
                text= 'Обучающая выборка',
                showlegend=False
            ),
            row=1, col=3
        )

        fig.add_trace(
            go.Scatter(
                x = x_axis,
                y = results['validation_1']['aucpr'],
                mode = "lines",
                name = "Проверочная выборка",
                marker = dict(color = 'rgba(255, 154, 0, 1)'),
                text= 'Проверочная выборка',
                showlegend=False
            ),
            row=1, col=3
        )

        # Update xaxis properties
        fig.update_xaxes(title_text="Iterations", row=1, col=1)
        fig.update_xaxes(title_text="Iterations", row=1, col=2)
        fig.update_xaxes(title_text="Iterations", row=1, col=3)

        # Update yaxis properties
        fig.update_yaxes(title_text="LogLoss", row=1, col=1)
        fig.update_yaxes(title_text="AUC", row=1, col=2)
        fig.update_yaxes(title_text="AUCPR", row=1, col=3)

        fig.update_layout(height=500, width=2000, title_text=f"{name_fold} фолд")

        # Интерактивный режим (много весит, отказался от него)
        #plotly.offline.plot(fig, auto_open=False, filename=f"./report/{name_save_fig}.html")
    
        fig.write_image(f"{path_base}.png")

    
    def create_report(self, name_domain:str, path_base:str):
        """[summary]

        Args:
            name_domain (str): [description]
            path_base (str): [description]
        """
        path_base = f'{path_base}'
        create_folder(path_base)

        self._plots_learning_curve(self.results_['eval_result_1'], f'{path_base }/my_fig_1', 'Первый')
        self._plots_learning_curve(self.results_['eval_result_2'], f'{path_base }/my_fig_2', 'Второй')
        self._plots_learning_curve(self.results_['eval_result_3'], f'{path_base }/my_fig_3', 'Третий')
        self._plots_learning_curve(self.results_['eval_result_4'], f'{path_base }/my_fig_4', 'Четвертый')
        self._plots_learning_curve(self.results_['eval_result_5'], f'{path_base }/my_fig_5', 'Пятый')

        df_fi = self.get_importances()
        df_fi.to_excel(f'{path_base}/feature_importance_{name_domain}.xlsx')

        with open(f'{path_base}/interactive_df.html','w', encoding='utf8') as f:
            f.write(df_html(df_fi))

        with open(f'{path_base}/interactive_df.html', 'r', encoding='utf8') as f:
            interactive_df = f.read()

        fig_1 = get_img_tag(f'{path_base }/my_fig_1.png')
        fig_2 = get_img_tag(f'{path_base }/my_fig_2.png')
        fig_3 = get_img_tag(f'{path_base }/my_fig_3.png')
        fig_4 = get_img_tag(f'{path_base }/my_fig_4.png')
        fig_5 = get_img_tag(f'{path_base }/my_fig_5.png')

        self.rocauc_plots(path_base)

        fig_6 = get_img_tag(f'{path_base }/rocauc_plots_train.png')
        fig_7 = get_img_tag(f'{path_base }/rocauc_plots_valid.png')

        m_gini_train = round((((
                   self.scores_dict['scores_1']['roc_train'] +
                   self.scores_dict['scores_2']['roc_train'] +
                   self.scores_dict['scores_3']['roc_train'] +
                   self.scores_dict['scores_4']['roc_train'] +
                   self.scores_dict['scores_5']['roc_train'])/5)*2 - 1)*100, 3)

        m_gini_valid = round((((
                   self.scores_dict['scores_1']['roc_valid'] +
                   self.scores_dict['scores_2']['roc_valid'] +
                   self.scores_dict['scores_3']['roc_valid'] +
                   self.scores_dict['scores_4']['roc_valid'] +
                   self.scores_dict['scores_5']['roc_valid'])/5)*2 - 1)*100, 3)
            

        h3 = '''<li>
                    <h3 class="caret" id="section1_3">1.3. Кривые обучения на фолдах</h3>
                        <ul class="nested">
                            ''' + fig_1 + '''
                            ''' + fig_2 + '''
                            ''' + fig_3 + '''
                            ''' + fig_4 + '''
                            ''' + fig_5 + '''
                        </ul>
                </li>'''


        html_string = f'''
            <!DOCTYPE html>
            <html lang="en">

            '''+start_report+'''

            '''+settings_style+f'''

        <body>
            <div>
                <h1>Автоматический отчет важностей факторов</h1>
                <hr>
                <h2>Домен: {name_domain}</h2>
                <hr>
            </div>
            
            <ul id="myUL">
                <li>
                    <h2 class="caret caret-down" id="section1">1. Важности факторов</h2>
                    <ul class="nested active">
                    
                        <li>
                            <h3 class="caret" id="section1_1">1.1. Цель документа</h3>
                            <ul class="nested">
                                <p>Данный отчет создан с целью анализа и вычислений важности факторов. 
                                Важности факторов расчитываются с помощью
                                кросс-валидации и модели XGBoost.</p>
                                <p>Тип вычисляемых важностей - GAIN.</p>
                                <p>Данный подход зарекомендовал себя как стабильный и надежный.
                                Такой подход позволяет получить стабильную оценку важностей факторов для 
                                дальнейшего выбора топ факторов для финальной модели.</p>
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_2">1.2. Общие параметры выборки</h3>
                            <ul class="nested">
                                <li>
                                    <p>Количество строк: {len(self.X_train)}</p>
                                    <p>Количество факторов: {len(self.features)}</p>
                                </li>
                            </ul>
                        </li>
                        
                        {h3}
                        
                        <li>
                            <h3 class="caret" id="section1_4">1.4. Метрики на фолдах</h3>
                            <ul class="nested">

                                <li>
                                    <p>Средний Gini на train: {m_gini_train} %</p>
                                    <p>Средний Gini на valid: {m_gini_valid} %</p>
                                </li>

                                ''' + fig_6 + '''
                                ''' + fig_7 + ''' 
                                
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_5">1.5. Таблица важностей</h3>
                            <ul class="nested">
                            ''' + interactive_df + '''
                            </ul>
                        </li>
                            
            ''' + end_report+ '''
            
        </body>

        </html>
        '''

        with open(f'{path_base}/final_report.html', 'w', encoding = 'utf8') as f:
            f.write(html_string)