import matplotlib.pyplot as plt
import shap

import pandas as pd
import numpy as np

from ..utils.html_style import settings_style, start_report, end_report
from ..utils.to_html import get_img_tag
from ..utils.folders import create_folder


class PlotShap:
    """
    """
    def __init__(self, model, sample):
        self.model = model
        self.sample = sample
    
    def fit_shap(self):
        """[summary]
        """
        self.shap_values = shap.TreeExplainer(self.model).shap_values(self.sample)
        
    def get_table_shap(self):
        
        feature_names = self.sample.columns
        shap_values = self.shap_values
        
        rf_resultX = pd.DataFrame(shap_values, columns = feature_names)
        
        vals = np.abs(rf_resultX.values).mean(0)

        shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                       columns=['col_name','imp_shap'])
        
        shap_importance = shap_importance.sort_values('imp_shap', ascending=False).reset_index(drop=True)
        
        return shap_importance


    def create_plot_shap(self, number_features:int=3, 
                         auto_size_plot: bool=True, show: bool=True, plot_type=None):
        """[summary]

        Args:
            number_features (int, optional): [description]. Defaults to 3.
            auto_size_plot (bool, optional): [description]. Defaults to True.
            show (bool, optional): [description]. Defaults to True.
            plot_type ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
     
        plot_shap = shap.summary_plot(
            self.shap_values, 
            self.sample,
            max_display=number_features, 
            auto_size_plot=auto_size_plot,
            show=show,
            plot_type=plot_type)

        return plot_shap
    
    
    def create_feature_shap(self, features):
        cols = list(self.sample.columns)
        
        index_features = []
        for f in features:
            index_features.append(cols.index(f))
        
        plot_shap = shap.summary_plot(
            self.shap_values[:, index_features], 
            self.sample[features],
            show=True,
            sort=False,
            auto_size_plot=True, 
            max_display = 100)
        
        return plot_shap
    
    
    def create_shap_report(self, path_base: str):
        """[summary]

        Args:
            path_base (str): [description]
        """
        create_folder(path_base)

        self.create_plot_shap(show=False, number_features=None)
        plt.savefig(f"{path_base}/sum_plot.png", bbox_inches="tight")
        plt.close()

        self.create_plot_shap(show=False, number_features=None, plot_type='bar')
        plt.savefig(f"{path_base}/list_plot.png", bbox_inches="tight")
        plt.close()
        
        fig_1 = get_img_tag(f'{path_base}/sum_plot.png')
        fig_2 = get_img_tag(f'{path_base}/list_plot.png')

        html_string = '''
            <!DOCTYPE html>
            <html lang="en">

        '''+start_report+'''

        '''+settings_style+f'''

        <body>
            <div>
                <h1>Автоматический отчет SHAP факторов</h1>
            </div>
            
            <ul id="myUL">
                <li>
                    <h2 class="caret caret-down" id="section1">1. Важности SHAP</h2>
                    <ul class="nested active">
                        
                        <li>
                            <h3 class="caret" id="section1_1">1.1. Общие параметры оцениваемой выборки</h3>
                            <ul class="nested">
                                <li>
                                    <p>Количество строк: {self.sample.shape[0]}</p>
                                    <p>Количество факторов: {self.sample.shape[1]}</p>
                                </li>
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_2">1.2. Расшифровка SHAP</h3>
                            <ul class="nested">
                                <p>SHAP позволяет получить инсайты из сложной модели, 
                                которые мы не можем просто так проинтерпертировать.</p>
                                <p>1. Одна точка в одном ряду - это один объект/клиент.</p>
                                <p>2. Чем толще линия, тем больше там наблюдений.</p>
                                <p>3. Чем более красная точка, тем больше значение этого признака.</p>
                                <p>4. Идеальный признак, разделяющий 2 класса: по одну сторону только 
                                красные точки, по другую сторону только синие точки от вертикальной линии.</p>
                                <p>5. Правая область отвечает за целевое действие (например, метка 1)</p>
                                <p>6. Чем точка правее, тем больший вклад она внесла в предсказание в алгоритме.</p>
                                <p>7. Пример: если справа расположены красные точки, то это означает, что объект с 
                                более высокими значениями этого признака склонен к целевой метке модели. (относится к классу 1)</p>
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_3">1.3. Summary plot</h3>
                            <ul class="nested">

                                ''' + fig_1 + '''
                                
                            </ul>
                        </li>

                        <li>
                            <h3 class="caret" id="section1_4">1.4. List plot</h3>
                            <ul class="nested">

                                ''' + fig_2 + '''
                                
                            </ul>
                        </li>

        ''' + end_report+ '''
           
            </body>

        </html>
        '''

        with open(f'{path_base}/shap_report.html', 'w', encoding = 'utf8') as f:
            f.write(html_string)