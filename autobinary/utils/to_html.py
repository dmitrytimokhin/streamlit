import base64

def get_img_tag(dir_img):
        """[summary]

        Args:
            dir_img ([type]): [description]

        Returns:
            [type]: [description]
        """
        data_uri = base64.b64encode(open(f'{dir_img}', 'rb').read()).decode('utf-8')
        img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
        return img_tag


base_html = """
            <!doctype html>
            <html>
                <head>
                    <meta http-equiv="Content-type" content="text/html; charset=utf-8">
                    <script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
                    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.16/css/jquery.dataTables.css">
                    <script type="text/javascript" src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.js"></script>
                </head>
                <body>%s
                    <script type="text/javascript">$(document).ready(function(){$('table').DataTable({
                        "pageLength": 10
                    });});
                    </script>
                </body>
            </html>
            """

def df_html(df):
    """HTML table with pagination and other goodies"""
    df_html = df.to_html()
    return base_html % df_html