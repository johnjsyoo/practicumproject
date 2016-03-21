import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import confusion_matrix


# Importing raw observations
df=pd.read_csv('raw_data_v3.csv')

""" Windows and Feature Selection """

# Dropping transient activity
df = df[df['activity_id'] != 0]

# Sorting by subject and timestamp
df = df.sort(['subject_id','timestamp'])

# Function for window size and features
def window_stats(dataframe, window_size):
    windows = np.array_split(dataframe, len(dataframe.index)/window_size)
    all_stats = []
    for window in windows:
        if window.iloc[0]['activity_id'] == window.iloc[window_size-1]['activity_id']:
            #ADD FEATURES HERE
            means = window.ix[:,3:34].mean()
            stds = window.ix[:,3:34].std()
            ranges = window.ix[:,3:34].apply(lambda x: x.max() - x.min())
            window_stats = pd.concat([pd.Series(window.iloc[0]['activity']),pd.Series(window.iloc[0]['subject_id']),
            means.astype(float),stds.astype(float),ranges.astype(float)],axis=0)
            all_stats.append(window_stats)
    d = pd.concat(all_stats,axis=1).T
    d.columns = ['activity','subject_id'] + [x + '_mean' for x in df.columns[3:34]] + \
    [x + '_std' for x in df.columns[3:34]] + [x + '_range' for x in df.columns[3:34]]
    return d

wd_data = window_stats(df, 500)

""" Features and Labels """

X = wd_data.ix[:,3:96].astype(float)
Y = wd_data.ix[:,0]

""" Cross Validation """

from sklearn.cross_validation import StratifiedKFold
n_folds=5
skf = StratifiedKFold(Y, n_folds)
 
""" Modeling """

# Fit models and report classification report and avg of fold scores
scores = []
models = {}
for name, clf, model in [('Logistic', LogisticRegression(), 'LogModel'),
                         ('Naive Bayes', GaussianNB(), 'NaiveBayesModel'),
                         ('Support Vector Classification', LinearSVC(C=1.0), 'SVMModel'),
                         ('Random Forest', RandomForestClassifier(n_estimators=100), 'RFModel'),
                         ('K Nearest Neighbors (N = 3)', KNeighborsClassifier(n_neighbors=3), 'KNeighborsModel'),
                         ('Bagging', BaggingClassifier(n_estimators=100), 'BaggingModel'),
                         ('Ada Boosting', AdaBoostClassifier(n_estimators=100), 'ADAModel'),
                         ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100,
                                                                          learning_rate=1.0,
                                                                          max_depth=1,
                                                                          random_state=4), 'GradientModel')]:
    clf_score = sum([clf.fit(X.iloc[train], Y.iloc[train]).score(X.iloc[test], Y.iloc[test]) for train, test in skf])/n_folds
    model = clf.fit(X.iloc[train], Y.iloc[train])
    print(name, ": \n", classification_report(Y, clf.predict(X)))
    scores.append((name, clf_score))
    models[name] = model

""" Neural Networks Modeling """

from sknn.mlp import Classifier, Layer

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100, pieces=2),
        Layer("Softmax")],
    learning_rate=0.001,
    n_iter=25)
nn.fit(X.iloc[train], Y.iloc[train])

""" Stratifying Part II """

from sklearn import metrics
from sklearn import cross_validation

def stratified_cv(X, Y, clf_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedKFold(Y, n_folds=n_folds, shuffle=shuffle)
    Y_pred = Y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X.iloc[ii], X.iloc[jj]
        Y_train = Y.iloc[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,Y_train)
        Y_pred[jj] = clf.predict(X_test)
    return Y_pred

""" Printing Accuracy Score """

print('Naive Bayes:                     {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, GaussianNB))))
print('Gradient Boosting Classifier:    {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, GradientBoostingClassifier))))
print('Support Vector Machine (SVM):    {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, LinearSVC, C=1.0))))
print('Random Forest Classifier:        {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, RandomForestClassifier, n_estimators=100))))
print('K Nearest Neighbor Classifier:   {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, KNeighborsClassifier, n_neighbors=3))))
print('Logistic Regression:             {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, LogisticRegression))))
print('Bagging:                         {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, BaggingClassifier))))
print('ADA Boosting:                    {:.2f}'.format(metrics.accuracy_score(Y, stratified_cv(X, Y, AdaBoostClassifier, n_estimators=100))))


""" Building a Confusion Matrix"""

naive_bayes_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, GaussianNB))
gradient_boosting_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, GradientBoostingClassifier))
svm_svc_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, LinearSVC, C=1.0))
rand_forest_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, RandomForestClassifier, n_estimators=100))
k_nearest_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, KNeighborsClassifier, n_neighbors=3))
log_regression_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, LogisticRegression))
bagging_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, BaggingClassifier))
ada_boost_conf_matrix = metrics.confusion_matrix(Y, stratified_cv(X, Y, AdaBoostClassifier, n_estimators=100))

conf_matrix = {
    1: {
        'matrix': naive_bayes_conf_matrix,
        'title': 'Naive Bayes',
    },
    2: {
        'matrix': gradient_boosting_conf_matrix,
        'title': 'Gradient Boosting',
    },
    3: {
        'matrix': svm_svc_conf_matrix,
        'title': 'Support Vector Machine',
    },
    4: {
        'matrix': rand_forest_conf_matrix,
        'title': 'Random Forest',
    },
    5: {
        'matrix': k_nearest_conf_matrix,
        'title': 'K Nearest Neighbors',
    },
    6: {
        'matrix': log_regression_conf_matrix,
        'title': 'Logistic Regression',
    },
    7: {
        'matrix': bagging_conf_matrix,
        'title': 'Bagging',
    },
    8: {
        'matrix': ada_boost_conf_matrix,
        'title': 'ADA Boosting',
    },
}

""" Plotting the Confusion Matrix """

import matplotlib.pyplot as plt
import seaborn as sns

target_names = np.unique(skf.y).tolist()     # Grabbing tick names

fix, ax = plt.subplots(figsize=(16, 12))
plt.suptitle('Confusion Matrix of Various Classifiers')
for ii, values in conf_matrix.items():
    matrix = values['matrix']
    title = values['title']
    plt.subplot(3, 3, ii)  # starts from 1
    plt.title(title);
    sns.heatmap(matrix, annot=True, fmt='');
    plt.xticks(range(len(target_names)), target_names, rotation=90)
    plt.yticks(range(len(target_names)), target_names)
