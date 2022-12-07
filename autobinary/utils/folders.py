import os

def create_folder(path_base: str):

    if os.path.exists(path_base) == True:
        print('Директория существует')
    else:
        os.mkdir(path_base)