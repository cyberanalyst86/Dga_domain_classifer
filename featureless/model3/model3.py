import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
BatchNormalization, Convolution1D, MaxPooling1D, concatenate, Dropout
from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
import pandas as pd
from keras.layers import Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from string import printable
from sklearn import model_selection

def lstm_convolution(max_data_len, embedding_dimension, max_vocab_len, lstm_output_size):

    model = Sequential()
    model.add(Embedding(input_dim=max_vocab_len, output_dim=embedding_dimension, input_length=max_data_len))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_output_size))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

#An epoch is one complete presentation of the data set to be learned to a learning machine

def run(max_num_of_epoch=20, number_of_fold=1, batch_size=128):

    labels = []
    domains = []

    #benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header_trial_1000.csv')
    benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header.csv')

    benign_df = benign_data[1:]

    domains += benign_df.Domain.tolist()
    labels += ['benign'] * len(benign_df.Domain)

    #dga_data = pd.read_csv("E://dga_data//botnet_with_header_trial_1000.csv")
    dga_data = pd.read_csv("E://dga_data//botnet_with_header.csv")
    dga_df = dga_data[1:]

    domains += dga_df.Domain.tolist()

    labels += ['dga'] * len(dga_df.Domain)

    domain_int_tokens = [[printable.index(x) + 1 for x in str(domain) if x in printable] for domain in domains]

    length = []

    for element in domain_int_tokens:

        length.append(len(element))

    max_data_len = max(length)

    print("max_data_len = ", max_data_len)

    # Step 2: Cut URL string at max_len or pad with zeros if shorter
    #max_length = np.max([len(str(element)) for element in domains])

    X = sequence.pad_sequences(domain_int_tokens, maxlen=100)

    # Step 3: Extract labels form df to numpy array
    label_data = [0 if element == 'benign' else 1 for element in labels]
    target = np.array(label_data)

    print('Matrix dimensions of X: ', X.shape, 'Vector dimension of target: ', target.shape)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, target, test_size=0.1, random_state=33)

    #y_train = np.expand_dims(y_train, axis=1)

    #y_test = np.expand_dims(y_test, axis=1)

    """--------------------------------------Build Model -----------------------------------------"""

    embedding_dimension = 128
    lstm_output_size = 32

    model = lstm_convolution(100, embedding_dimension, 100, lstm_output_size)

    print("train_cnn_lstm_model")

    model.fit(x_train, y_train, batch_size=batch_size, epochs=max_num_of_epoch)

    print("evaluate_cnn_lstm_model")

    score = model.evaluate(x_test, y_test)
    print('Test accuarcy: %0.4f%%' % (score[1] * 100))

    """--------------------------------------Save Model -----------------------------------------"""

    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_lstm_model.h5")
    print("Saved model to disk")

    """--------------------------------------Load Model -----------------------------------------"""
    """
    #loading model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    """

run()