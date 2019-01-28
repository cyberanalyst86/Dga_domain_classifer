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

def  cnn_lstm_model(max_features, max_length):

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation ='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model

#An epoch is one complete presentation of the data set to be learned to a learning machine

def run(max_num_of_epoch=20, batch_size=128):

    labels = []
    domains = []

    benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header.csv')

    benign_df = benign_data[1:]

    domains += benign_df.Domain.tolist()
    labels += ['benign'] * len(benign_df.Domain)

    dga_data = pd.read_csv("E://dga_data//botnet_with_header.csv")
    dga_df = dga_data[1:]

    domains += dga_df.Domain.tolist()
    labels += ['dga'] * len(dga_df.Domain)

    print("domains size = ",  len(domains))
    print("labels size = ", len(labels))

    # Generate dictionary of valid characters
    character_dictionary = {}

    for index, element in enumerate(set(''.join(map(str, domains)))):
        index += 1  # to remove int 0
        character_dictionary.update({element: index})

    max_features = len(character_dictionary) + 1

    max_length = np.max([len(str(element)) for element in domains])

    # Convert characters to int and pad

    int_list_of_list = []

    for element in domains:

        int_list = []

        for char in str(element):
            int_list.append(character_dictionary[char])

        int_list_of_list.append(int_list)

    int_data = int_list_of_list

    print("int_data size = ", len(int_data))

    int_data_padded = sequence.pad_sequences(int_data, maxlen=max_length, value=0.)

    # Convert labels to 0-1
    label_data = [0 if element == 'benign' else 1 for element in labels]

    print("size of labels = ", len(labels))

    x_train, x_test, y_train, y_test, label_train, label_test = train_test_split(int_data_padded, label_data, labels,test_size=0.1)

    print("x_train size = ", len(x_train))
    print("y_train size = ", len(y_train))
    print("x_test size = ", len(x_test))
    print("y_test size = ", len(y_test))

    model = cnn_lstm_model(max_features, max_length)

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

    """--------------------------------------Save Model -----------------------------------------

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