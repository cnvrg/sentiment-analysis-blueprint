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

import keras
import tensorflow as tf
# DataFrame
import pandas as pd
# Matplot
import matplotlib.pyplot as plt
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
# Keras
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
# nltk
from nltk.stem import SnowballStemmer
# Word2vec
import gensim
# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import pickle
import argparse

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

print('gpuname: ', tf.config.experimental.list_physical_devices('GPU'))

def read_dataset(input_filename):
        dataset_columns = ["target", "timestamp", "datetime", "query", "user", "text"]
        dataset_encoding = "ISO-8859-1"
        return pd.read_csv(input_filename, encoding=dataset_encoding, names=dataset_columns)

def decode_sentiment(label):
    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
    if type(label) == str:
        print(label)
        label = int(label)
    return decode_map[label]

def preprocess(text, stem=False):
    # Remove link, user and special characters
    text_cleaning_regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_regex, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if stem:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    return " ".join(tokens)

def split_train_test_data(df, train_size):
    # Split train and test
    df_train, df_test = train_test_split(df, test_size=1 - train_size, random_state=42)
    return df_train, df_test

def create_documents(df_train):
    return [_text.split() for _text in df_train.text]

def word_2_vector(documents, w2v_size, w2v_window, w2v_epochs_val, w2v_min_count):
    # Word2vec
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=w2v_size,
                                                window=w2v_window,
                                                min_count=w2v_min_count,
                                                workers=8)
    w2v_model.build_vocab(documents)
    w2v_model.train(documents, total_examples=len(documents), epochs=w2v_epochs_val)
    return w2v_model

def tokenize_text(df_train):
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.text)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size

def padding_sequences(df_train, df_test, seq_len):
    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=seq_len)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=seq_len)
    return x_train, x_test

def label_encoding(df_train):
    # Label encoder
    labels = df_train.target.unique().tolist()
    labels.append("NEUTRAL")

    encoder = LabelEncoder()
    encoder.fit(df_train.target.tolist())
    return encoder, labels

def prepare_labels(df_train, df_test, encoder):
    y_train = encoder.transform(df_train.target.tolist())
    y_test = encoder.transform(df_test.target.tolist())
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return y_train, y_test

def create_embedding_layer(vocab_size, tokenizer, w2v_model, w2v_size, seq_len):
    # Embedding layer
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    embedding_layer = Embedding(vocab_size, w2v_size, weights=[embedding_matrix], input_length=seq_len,
                                trainable=False)
    return embedding_layer

def build_model(embedding_layer):
    # Build model
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def compile_model(model):
    # Compile model
    model.compile(loss='binary_crossentropy',
                optimizer="adam",
                metrics=['accuracy'])

def get_callbacks():
    return [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

def fit_model(model, x_train, y_train, epochs_val, batch_size_val, callbacks):
        history = model.fit(x_train, y_train,
                        batch_size=batch_size_val,
                        epochs=epochs_val,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=callbacks)
        return history

def get_score(model, batch_size_val):
    score = model.evaluate(x_test, y_test, batch_size=batch_size_val)
    return score

def get_performace_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return acc, val_acc, loss, val_loss

def plot_loss_graph(acc, val_acc, loss, val_loss):
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def save_tokenizer(file_dir, output_token_file, tokenizer):
    with open(file_dir + f'/{output_token_file}', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_model(file_dir, output_model_file):
    model.save(file_dir + '/{}'.format(output_model_file))

if __name__ == "__main__":
    # Set log
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description="""Preprocessor""")

    parser.add_argument('-f', '--input_filename', action='store', dest='input_filename', required=True,
                        help="""string. csv twitter labeled train data file""")

    parser.add_argument('-l', '--local_dir', action='store', dest='local_dir', default=cnvrg_workdir, required=False,
                        help="""file_dir to store model file""")

    parser.add_argument('-o', '--output_token_file', action='store', dest='output_token_file', default='tokenizer.pickle',
                        required=False,
                        help="""string. filename for saving the tokenizer""")

    parser.add_argument('-m', '--output_model_file', action='store', dest='output_model_file',
                        default='model.h5', required=False,
                        help="""string. filename for saving the model""")

    parser.add_argument('-t', '--text_column', action='store', dest='text_column', default='text', required=False,
                        help="""string. name of text column""")

    parser.add_argument('-s', '--label_column', action='store', dest='label_column', default='sentiment', required=False,
                        help="""string. name of label column""")

    parser.add_argument('--epochs_val', action='store', dest='epochs_val', default=64, required=False,
                        help="""int. number of training epochs to run""")

    parser.add_argument('--w2v_epochs_val', action='store', dest='w2v_epochs_val', default=32, required=False,
                        help="""int. number of training epochs to run""")

    parser.add_argument('--batch_size_val', action='store', dest='batch_size_val', default=256, required=False,
                        help="""int. size of each batch to train on""")

    parser.add_argument('--train_size', action='store', dest='train_size', default=0.8, required=False,
                        help="""Fraction of dataset to be assigned as training data""")

    parser.add_argument('--w2v_size', action='store', dest='w2v_size', default=300, required=False,
                        help="""vector size parameter in word to vector model """)

    parser.add_argument('--w2v_window', action='store', dest='w2v_window', default=7, required=False,
                        help="""window size parameter in word to vector model """)

    parser.add_argument('--w2v_min_count', action='store', dest='w2v_min_count', default=10, required=False,
                        help="""min count parameter in word to vector model """)

    parser.add_argument('--seq_len', action='store', dest='seq_len', default=300, required=False,
                        help="""seq len for NLP model""")

    args = parser.parse_args()
    output_token_file = args.output_token_file
    output_model_file = args.output_model_file
    text_column = args.text_column
    label_column = args.label_column
    epochs_val = int(args.epochs_val)
    batch_size_val = int(args.batch_size_val)
    input_filename = args.input_filename
    file_dir = args.local_dir
    train_size = args.train_size
    w2v_size = int(args.w2v_size)
    w2v_window = int(args.w2v_window)
    w2v_epochs_val = int(args.w2v_epochs_val)
    w2v_min_count = int(args.w2v_min_count)
    seq_len = int(args.seq_len)

    df = read_dataset(input_filename)
    df.target = df.target.apply(lambda x: decode_sentiment(x))

    # Pre-process dataset
    stemmer = SnowballStemmer("english")
    df.text = df.text.apply(lambda x: preprocess(x))

    df_train, df_test = split_train_test_data(df, train_size)
    documents = create_documents(df_train)
    w2v_model = word_2_vector(documents, w2v_size, w2v_window, w2v_epochs_val, w2v_min_count)
    tokenizer, vocab_size = tokenize_text(df_train)

    print("FINISHED WORD2VEC")
    x_train, x_test = padding_sequences(df_train, df_test, seq_len)
    print("FINISHED PADDING")

    encoder, labels = label_encoding(df_train)
    y_train, y_test = prepare_labels(df_train, df_test, encoder)
    print("FINISHED TRAINING")

    embedding_layer = create_embedding_layer(vocab_size, tokenizer, w2v_model, w2v_size, seq_len)
    
    print("BUILDING MODEL")
    model = build_model(embedding_layer)
    compile_model(model)

    # Callbacks
    callbacks = get_callbacks()

    history = fit_model(model, x_train, y_train, epochs_val, batch_size_val, callbacks)
    score = get_score(model, batch_size_val)
    print()
    print("ACCURACY:", score[1])
    print("LOSS:", score[0])

    acc, val_acc, loss, val_loss = get_performace_metrics(history)
    plot_loss_graph(acc, val_acc, loss, val_loss)

    save_tokenizer(file_dir, output_token_file, tokenizer)

    save_model(file_dir, output_model_file)