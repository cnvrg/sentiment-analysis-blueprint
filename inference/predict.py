# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import os
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pathlib
import sys
scripts_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))
from prerun import download_model_files
download_model_files()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
if os.path.exists('/input/train'):  # If the model artifacts are from train
    model_path = '/input/train/model.h5'
    tokenizer_path = '/input/train/tokenizer.pickle'
else:
    # TODO insert preprocess to download models
    scripts_dir = pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(scripts_dir, 'cpu_model.h5')
    tokenizer_path = os.path.join(scripts_dir, 'cpu_tokenizer.pickle')

model = keras.models.load_model(model_path, compile=False)
print("loaded model")
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

sentiment_label = ['positive', 'negative']

def predict(data):
    sentiment_label = ['negative', 'positive']
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([data['text']]), maxlen=300)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = sentiment_label[int(score.round().item())]
    return {"label": label, "score": float(score)}

