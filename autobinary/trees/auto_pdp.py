import matplotlib.pyplot as plt
from pdpbox import pdp
import PIL.Image as pil

import pandas as pd
import numpy as np

from ..utils.html_style import settings_style, start_report, end_report
from ..utils.to_html import get_img_tag
from ..utils.folders import create_folder

class PlotPDP:
    """
    """
    def __init__(self,model: object, X: pd.DataFrame, main_features: list):
        self.model = model
        self.X = X
        self.model_features = list(self.X.columns)
        self.main_features = main_features
        
    def create_feature_plot(self, save: bool, frac_to_plot: float=0.1, n_jobs=-1, path: str='./pdp_ice_plots'):
        
        if save:
                create_folder(path)
                
        self.frac_to_plot = int(frac_to_plot*len(self.X))
        
        for i in self.main_features:
            print(i)
            pdp_iso = pdp.pdp_isolate(model=self.model, dataset=self.X, model_features=self.model_features, feature=i,)
            pdp.pdp_plot(pdp_iso, i, plot_lines=True, frac_to_plot=frac_to_plot, x_quantile=True, center=True)

            if save:
                plt.savefig('{}/PDP_{}.png'.format(path,i), bbox_inches="tight")
            plt.show()
        
    def create_interact_plot(self, features: list, save: bool, path: str='./pdp_ice_plots'):
        inter = pdp.pdp_interact(model=self.model, dataset=self.X, model_features=self.model_features,features=features)
        
        fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter,
                                          feature_names=features,
                                          plot_type='grid',
                                          x_quantile=True,
                                          plot_pdp=True)
        if save:
            try:
                fig.savefig('{}/PDP_interact_{}.png'.format(path,str(features)), bbox_inches="tight")
            except:
                print('Директория отсутствует, запустите функцию create_feature_plot с параметров save=True')
                
        print('Интерпретация: Как факторы ', str(features), 'совместо влияют на предсказание. Чем ярчке (желтая) ячейка, тем сильнее влияние в совокупности: Для классификации - растет вероятность отнесения к целевой метке класса; Для регрессии - растет значение предсказания. Чем темнее (фиолетовая) ячейка - обратная ситуация.')

    def create_pdp_report(self, path_base: str='./pdp_ice_plots'):
        
        print('Загрузка всех сохраненных изображений','\n')
        
        try:
            figs = {i: pil.open('{}/PDP_{}.png'.format(path_base,i)) for i in self.main_features}
            min_shape = sorted([(np.sum(i.size), i.size ) for i in figs.values()])[0][1]
            imgs_comb = np.vstack((np.asarray( i.resize(min_shape)) for i in figs.values()))
            pil.fromarray(imgs_comb).save('{}/PDP_full.png'.format(path_base))
            
        except:
            print('Одно из изображений отсутствует, запустите функцию create_feature_plot с параметров save=True')
        
        print('Загрузка завершена.','\n')
        
        fig = get_img_tag('{}/PDP_full.png'.format(path_base))     
        
        html_string = '''
            <!DOCTYPE html>
            <html lang="en">

        '''+ start_report +'''

        '''+ settings_style +f'''

        <body>
            <div>
                <h1>Автоматический отчет PDP факторов</h1>
            </div>
            
            <ul id="myUL">
                <li>
                    <h2 class="caret caret-down" id="section1">1. Интерпретация факторов через PDP-ICE plots анализ</h2>
                    <ul class="nested active">
                        
                        <li>
                            <h3 class="caret" id="section1_1">1.1. Общие параметры оцениваемой выборки</h3>
                            <ul class="nested">
                                <li>
                                    <p>Количество строк: {self.X.shape[0]}</p>
                                    <p>Количество факторов: {self.X.shape[1]}</p>
                                    <p>Доля наблюдений в общем количестве для анализа кривых: {round((self.frac_to_plot/self.X.shape[0])*100)}% <p>
                                </li>
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_2">1.2. Расшифровка PDP-ICE</h3>
                            <ul class="nested">
                                <p>PDP-ICE plot анализ позволяет получить инсайты из сложной модели, 
                                которые мы не можем просто так проинтерпертировать.</p>
                                <p>1. Красная пунктирная линия - неопределенность (нулевое значчение).<p>
                                <p>2. Одна синяя линия - это один объект/клиент.</p>
                                <p>3. Одна желтая линия - это среднее по всем наблюдениям/клиентам.<p>
                                <p>4. На оси абцисс (Х) ранжирования признака от меньшего к большему.</p>
                                <p>5. На оси ординат (Y) Для классификации: выше/ниже нулевого значения - рост/ падение вероятности отнесения к целевой метке модели (класс 1);
                                Для регрессии: выше/ ниже нулевого значения - рост/ падение значения целевой переменной.<p>
                                <p>6. Идеальный признак: монотонное изменение вероятности отнесения к классу, при увеличении значения фактора.</p>
                                <p>7. Пример: если большинство синих линий, а значит и желтая линия монотонно возрастают по мере увеличения признака,
                                то объект склонен к целевой метке модели (относится к классу 1).</p>
                            </ul>
                        </li>
                        
                        <li>
                            <h3 class="caret" id="section1_3">1.3. PDP-ICE plots</h3>
                            <ul class="nested">
                                ''' + fig + '''
                                
                            </ul>
                        </li>

        ''' + end_report + '''
           
            </body>

        </html>
        '''

        with open(f'{path_base}/pdp_report.html', 'w', encoding = 'utf8') as f:
            f.write(html_string)
            
        print('Отчет сгенерирован.')