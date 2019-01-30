import numpy as np
import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
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

    new_dga_data = dga_data.drop(columns = "Label")

    new_dga_data["Label"] = "dga"

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

    x = all_domains[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values
    y = np.array(all_domains['Label'].tolist())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(x_train.shape))
    print("Test data   : {}".format(x_test.shape))
    print("Train labels: {}".format(y_train.shape))
    print("Test labels : {}".format(y_test.shape))

    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')


    """---------------------------------------Model Definition------------------------------------------"""
    LR = LogisticRegression(random_state=1)
    RF = RandomForestClassifier(bootstrap=True, max_depth=None, class_weight="balanced", min_samples_leaf=1,
                                  min_samples_split=1.0, n_estimators=1500, n_jobs=40, oob_score=False,
                                  random_state=1, verbose=1)
    GNB = GaussianNB()
    ETR = ExtraTreesClassifier()

    """---------------------------------------Fit ------------------------------------------"""
    all_model = [LR, RF, GNB, ETR]

    for model in all_model:

        model.fit(x_train, y_train)

    ES = VotingClassifier(estimators=[('LR', LR), ('RF', RF), ('GNB', GNB), ('ETR', ETR)], voting='soft')

    file = open("model_accuracy.txt", "a")

    for model, label in zip([LR, RF, GNB, ETR, ES], ['Logistic Regression', 'Random Forest', 'naive Bayes',
                                                           'Extra Tree', 'Ensemble']):

        scores = cross_val_score(model, x_test, y_test, cv=5, scoring='accuracy')
        print("Accuracy: %0.6f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

        file.write("Accuracy: %0.6f (+/- %0.2f) [%s] \n" % (scores.mean(), scores.std(), label))

run()




