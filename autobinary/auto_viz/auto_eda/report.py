import os

path_base = './reports'

if os.path.exists(path_base ) == True:
    print('Директория существует')
else:
    os.mkdir(path_base )


from .wing_eda import WingsOfEvidence
from .main import BaseEda


class Report:
    def __init__(self, df, num_columns, cat_columns, target):
        """[summary]

        Args:
            df ([type]): [description]
            num_columns ([type]): [description]
            cat_columns ([type]): [description]
        """
        self.df = df.copy()
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.all_columns = num_columns+cat_columns
        self.target = target

    def create_report(self):

        eda = BaseEda(self.df)

        desc_table = eda.base_overview().to_html()
        eda.visual_null(path=path_base)

        woe = WingsOfEvidence(columns_to_apply='all',
                              n_initial=20,
                              n_target=5,
                              only_values=False,
                              verbose=False)

        woe.fit(self.df[self.all_columns], self.df[self.target])   

        error_columns = woe.error_columns

        string_rep = ''
        counter = 0

        for col in self.all_columns:
            if col not in error_columns:
                counter+=1
                df_temp = woe.fitted_wing[col].get_wing_agg(only_clear=False).to_html()
    
                plot_woe = woe.fitted_wing[col].plot_woe()
                plot_woe.write_image(path_base + f"/1_{col}.png")
    
                plot_gini = woe.fitted_wing[col].display_gini()
                plot_gini.write_image(path_base + f"/2_{col}.png")

                if col in self.num_columns:
                    eda.num_target_plot_advanced(col, self.target, quant = -1.0, path=path_base)
                    eda.num_target_plot_advanced(col, self.target, quant = 0.05, path=path_base)

                    eda.num_target_plot(col, self.target, path=path_base)
                    eda.num_plot(col, path=path_base)

                    string_rep = string_rep + f'''
                                       <p style="background:#666;padding:10px;color:#fff">
                                       {counter}. Результаты woe для фактора {col}
                                       </p>
                                           <img src=1_{col}.png>
                                       </p>
                                       <p>
                                           <img src=2_{col}.png>
                                       </p>
                                       <p>
                                       {df_temp}
                                       </p>
                                       <p>
                                           <img src={col}-1.0_ft.png>
                                           <img src={col}0.05_ft.png>
                                       </p>
                                       <p>
                                           <img src={col}_num_plot.png>
                                           <img src={col}_num_target_plot.png>
                                       </p>
                                       '''
                else:
                    eda.cat_plot(col, path=path_base)

                    string_rep = string_rep + f'''
                                       <p style="background:#666;padding:10px;color:#fff">
                                       {counter}. Результаты woe для фактора {col}
                                       </p>
                                           <img src=1_{col}.png>
                                       </p>
                                       <p>
                                           <img src=2_{col}.png>
                                       </p>
                                       <p>
                                       {df_temp}
                                       </p>
                                       <p>
                                           <img src = {col}_cat_plot.png>
                                       </p>
                                       '''      

        template = f'''
                    <html>
                        <head>
                            <title>Заголовок документа </title>
                        <head>
                        <body>
                            <h1 style="background:#666;padding:10px;color:#fff">
                            Отчет EDA
                            </h1>
                            <p>
                                Количество строк: {self.df.shape[0]}
                                Количество факторов: {self.df.shape[1]}

                                Таблица базовых статистик:
                            </p>
                            <p>
                                {desc_table}
                            </p>
                            <p>
                                Визуализация пропущенных значений:
                            </p>
                            <p>
                                <img src=visual_null.png>
                            </p>
                            <p>
                                {string_rep}
                            </p>
                        </body>
                    </html>
                    '''

        with open(f'{path_base}/report_woe.html','w+') as file:
            file.write(template)



            