import requests
import os
import pathlib

BASE_FOLDER_URL = "https://blueprints-data.s3.us-west-2.amazonaws.com/model_files/sentiment_analysis/cpu/"
FILES = ['cpu_model.h5',
         'cpu_tokenizer.pickle']


def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(current_dir + f'/{f}') and not os.path.exists('/input/train/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            with open(f, "wb") as fb:
                fb.write(response.content)
