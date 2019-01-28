import numpy as np
import pickle
#from keras.preprocessing import sequence
#from keras.utils import to_categorical
#from keras.models import Sequential, Model, model_from_json, load_model
#from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
#BatchNormalization, Convolution1D, MaxPooling1D, concatenate, Dropout
#from keras.layers.core import Dense, Dropout, Activation, Lambda, Flatten
import sklearn
from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from keras import backend as K
import pandas as pd
#from keras.layers import Activation, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from sklearn.model_selection import cross_val_score
#from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sklearn.ensemble

def entropy(domain):
    p, lns = Counter(domain), float(len(domain))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

def run():

    word_dataframe = pd.read_csv('help/words.txt', names=['word'], header=None, dtype={'word': np.str},
                                 encoding='utf-8')
    word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
    word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
    word_dataframe = word_dataframe.dropna()
    word_dataframe = word_dataframe.drop_duplicates()

    benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header_trial_1000.csv')
    #benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header.csv')

    dga_data = pd.read_csv("E://dga_data//botnet_with_header_trial_1000.csv")
    #dga_data = pd.read_csv("E://dga_data//botnet_with_header.csv")

    new_benign_data = benign_data.drop(columns = 'Label')

    new_benign_data["Label"] = "benign"

    #print(new_benign_data)

    new_dga_data = dga_data.drop(columns = "Label")

    new_dga_data["Label"] = "dga"

    #print(new_dga_data)

    all_domains = pd.concat([new_benign_data, new_dga_data])

    #create panda column "length'
    all_domains['length'] = [len(domain) for domain in all_domains['Domain']]

    # create panda column "entropy'
    all_domains['entropy'] = [entropy(domain) for domain in all_domains['Domain']]

    alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
    #counts_matrix = alexa_vc.fit_transform(dataframe_dict['alexa']['domain'])
    counts_matrix = alexa_vc.fit_transform(benign_data['Domain'])
    alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

    dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
    counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
    dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())

    all_domains['alexa_grams'] = alexa_counts * alexa_vc.transform(all_domains['Domain']).T
    all_domains['word_grams'] = dict_counts * dict_vc.transform(all_domains['Domain']).T
    all_domains['diff'] = all_domains['alexa_grams'] - all_domains['word_grams']

    X = all_domains[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values
    y = np.array(all_domains['Label'].tolist())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(X_train.shape))
    print("Test data   : {}".format(X_test.shape))
    print("Train labels: {}".format(y_train.shape))
    print("Test labels : {}".format(y_test.shape))

    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')

    clf1 = LogisticRegression(random_state=1)
    clf1.fit(X_train, y_train)
    clf2 = RandomForestClassifier(bootstrap=True, max_depth=None, class_weight="balanced", min_samples_leaf=1,
                                  min_samples_split=1.0, n_estimators=1500, n_jobs=40, oob_score=False,
                                  random_state=1, verbose=1)
    clf2.fit(X_train, y_train)
    clf3 = GaussianNB()
    clf3.fit(X_train, y_train)
    clf4 = ExtraTreesClassifier()
    clf4.fit(X_train, y_train)


    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('etr', clf4)], voting='soft')

    file = open("model_accuracy.txt", "a")

    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                           'Extra Tree', 'Ensemble']):
        scores = cross_val_score(clf, X_test, y_test, cv=5, scoring='accuracy')
        print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

        file.write("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    """
    from sklearn.metrics import confusion_matrix
    y_pred = clf2.predict(X_test)
    labels = ['benign', 'dga']
    cm = confusion_matrix(y_test, y_pred, labels)

    def plot_cm(cm, labels):
        percent = (cm * 100.0) / np.array(np.matrix(cm.sum(axis=1)).T)  # Derp, I'm sure there's a better way
        print('Confusion Matrix Stats')
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent[i][j]), cm[i][j], cm[i].sum()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(b=False)
        cax = ax.matshow(percent, cmap='coolwarm')
        pylab.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        pylab.xlabel('Predicted')
        pylab.ylabel('True')
        pylab.show()

    plot_cm(cm, labels)
    """
run()




