#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import csv
import cv2
import os
import h5py
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from collections import Counter
import math
from sklearn.feature_extraction.text import CountVectorizer

# filter all the warnings
import warnings

warnings.filterwarnings('ignore')

def entropy(domain):
    p, lns = Counter(domain), float(len(domain))
    return -sum(count / lns * math.log(count / lns, 2) for count in p.values())

"""------------------------------------preprocess data------------------------------------"""

word_dataframe = pd.read_csv('help/words.txt', names=['word'], header=None, dtype={'word': np.str},
                                 encoding='utf-8')
word_dataframe = word_dataframe[word_dataframe['word'].map(lambda x: str(x).isalpha())]
word_dataframe = word_dataframe.applymap(lambda x: str(x).strip().lower())
word_dataframe = word_dataframe.dropna()
word_dataframe = word_dataframe.drop_duplicates()

benign_data = pd.read_csv('E://dga_data//alexa_1mil_with_header_trial_1000.csv')
dga_data = pd.read_csv("E://dga_data//botnet_with_header_trial_1000.csv")

new_benign_data = benign_data.drop(columns='Label')
new_benign_data["Label"] = "benign"

new_dga_data = dga_data.drop(columns="Label")
new_dga_data["Label"] = "dga"

all_domains = pd.concat([new_benign_data, new_dga_data])

# create panda column "length'
all_domains['length'] = [len(domain) for domain in all_domains['Domain']]

# create panda column "entropy'
all_domains['entropy'] = [entropy(domain) for domain in all_domains['Domain']]

alexa_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-4, max_df=1.0)
counts_matrix = alexa_vc.fit_transform(benign_data['Domain'])
alexa_counts = np.log10(counts_matrix.sum(axis=0).getA1())

dict_vc = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1e-5, max_df=1.0)
counts_matrix = dict_vc.fit_transform(word_dataframe['word'])
dict_counts = np.log10(counts_matrix.sum(axis=0).getA1())

# create panda column 'alexa_grams'
all_domains['alexa_grams'] = alexa_counts * alexa_vc.transform(all_domains['Domain']).T

# create panda column 'word_grams'
all_domains['word_grams'] = dict_counts * dict_vc.transform(all_domains['Domain']).T

# create panda column 'diff'
all_domains['diff'] = all_domains['alexa_grams'] - all_domains['word_grams']

x = all_domains[['length', 'entropy', 'alexa_grams', 'word_grams', 'diff']].values
y = np.array(all_domains['Label'].tolist())

# verify the shape of the feature vector and labels
print("original data information")
print("features shape: {}".format(x.shape))
print("labels shape: {}".format(y.shape))
print("start_training")


"""------------------------model parameters------------------------------------"""
seed = 9
test_size = 0.10
num_trees = 100

"""------------------------create model----------------------------------------"""
# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT'
               '', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))

# variables to hold the results and names
results = []
model_name = []
scoring = "accuracy"



"""------------------split the training and testing data----------------------"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

print("splitted data information")
print("Train data  : {}".format(x_train.shape))
print("Test data   : {}".format(x_test.shape))
print("Train labels: {}".format(y_train.shape))
print("Test labels : {}".format(y_test.shape))


"""-----------------Training Performance----------------------"""

print("Training Performance:")

# 10-fold cross validation
for name, model in models:
    number_of_fold = KFold(n_splits=10, random_state=7)
    cross_validation_results= cross_val_score(model, x_train, y_train, cv=number_of_fold, scoring=scoring)
    results.append(cross_validation_results)
    model_name.append(name)
    print("%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std()))

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
axis = fig.add_subplot(111)
pyplot.boxplot(results)
axis.set_xticklabels(model_name)
pyplot.show()

"""-----------------Training Performance----------------------"""

print("Testing Performance:")

for name, model in models:

    cross_validation_results= cross_val_score(model, x_test, y_test, cv=number_of_fold, scoring=scoring)
    results.append(cross_validation_results)
    model_name.append(name)
    print("%s: %f (%f)" % (name, cross_validation_results.mean(), cross_validation_results.std()))