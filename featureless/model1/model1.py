import numpy as np

from keras.preprocessing.text import Tokenizer
from random import shuffle
import json
import csv
from sklearn import model_selection
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Activation, Conv1D, MaxPooling1D

char_map = {'0': 0,'1': 1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,
    '.': 10, 'a': 11,'b': 12,'c': 13,'d': 14,'e': 15,'f': 16,'g': 17,'h': 18,'i': 19,
    'j': 20,'k': 21,'l': 22,'m': 23,'n': 24,'o': 25,'p': 26,'q': 27,'r': 28,'s': 29,
    't': 30,'u': 31,'v': 32,'w': 33,'x': 34,'y': 35,'z': 36,'-': 37,'_': 38}

top_level_domain = ['.com', '.net', '.biz', '.ru', '.org', '.co.uk', '.info', '.cc', '.ws', '.cn']

pad_number = 25

def convert_to_int_seq(text):

    int_seq = []
    for char in text:
        int_seq.append(char_map[char])
    return int_seq

def pad_int_seq(int_seq, max_len):

    global pad_number

    if len(int_seq) > max_len:

        return int_seq[:max_len]

    #word[:2]  # character from the beginning to position 2 (excluded)

    for i in range(len(int_seq), max_len):

        int_seq.append(pad_number)

    return int_seq

def remove_tld(domain):
    global top_level_domain
    for tld in top_level_domain:
        if tld in domain and tld is not None:
            return domain.replace(tld, '')

    return None

def read_csv_file():

    dga_csv = open('E://dga_data//botnet_with_header.csv', encoding="utf8")
    dga_data = csv.reader(dga_csv)
    next(dga_data)

    benign_csv = open('E://dga_data//alexa_1mil_with_header_reduced.csv', encoding="utf8")
    benign_data = csv.reader(benign_csv)
    next(benign_data)

    return dga_data, benign_data

def pre_process_data():

    dga_data, benign_data = read_csv_file()

    domain_name_list = []
    domain_name_label_dictionary = {}

    """ -------------------------- process dga_data ------------------------"""
    for row in dga_data:

        domain = row[0]  # first index of each row == domain

        domain_name_list.append(domain)

        domain_name_label_dictionary[domain] = 1

    """ -------------------------- process dga_data ------------------------"""
    for row in benign_data:

        domain = row[0]  # first index of each row == domain

        domain_name_list.append(domain)

        domain_name_label_dictionary[domain] = 0

    shuffle(domain_name_list)

    max_length = np.max([len(str(element)) for element in domain_name_list])

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    split_ratio = 0.1

    split_factor = split_ratio * len(domain_name_list)

    iteration_index = 0

    for element in domain_name_list:

        int_seq = convert_to_int_seq(element)
        padded_int_seq = pad_int_seq(int_seq, max_length)

        if iteration_index < split_factor:
            x_test.append(padded_int_seq)
            y_test.append(domain_name_label_dictionary[element])
        else:
            x_train.append(padded_int_seq)
            y_train.append(domain_name_label_dictionary[element])

        iteration_index+=1

    return (x_train, y_train), (x_test, y_test), max_length

def cnn_lstm_model(max_features, max_length):

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation ='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model

def run(max_num_of_epoch=20, batch_size=128):

    max_features = len(char_map)
    print('Load data')

    (x_train, y_train), (x_test, y_test), max_length = pre_process_data()

    model = cnn_lstm_model(max_features, max_length)

    print("train_cnn_lstm_model")

    model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=max_num_of_epoch)

    print("evaluate_cnn_lstm_model")

    score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)
    print('Test accuarcy: %0.4f%%' % (score[1] * 100))

    """--------------------------------------Save Model -----------------------------------------"""
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("cnn_lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_lstm_model.h5")
    print("Saved model to disk")
    """
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
