__author__ = 'John'

import pandas as pd
import numpy as np

df = pd.read_csv('raw_data_v2.csv')
df = df.rename(columns={'35-acticity' : '35-activity'})

def window_stats(dataframe, window_size):
    windows = np.array_split(dataframe, len(df.index)/window_size)
    all_stats = []
    for window in windows:
        means = window.ix[:,3:34].mean()
        stds = window.ix[:,3:34].std()
        ranges = window.ix[:,3:34].apply(lambda x: x.max() - x.min())
        window_stats = pd.concat([pd.Series(window.iloc[0]['35-activity']),means,stds,ranges],axis=0)
        all_stats.append(window_stats)
    return pd.concat(all_stats,axis=1).T

window_stats(raw_data, 100)

grouped = df.groupby(['34-subject_id', '2-activityid'])
len(grouped)
grouped.count()

data = pd.read_csv('1000.csv')

list(data.columns.values)[2:33]
list(data.columns.values)[34:64]
list(data.columns.values)[65:95]

## SVMs

from sklearn import svm

## Logistic Regression
from sklearn.linear_model import LogisticRegression


## Random Forests
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 1000)            # Tuning Parameters, see Documentation
forest = forest.fit(train_data[0::,1::],train_data[0::,0])      # Fix this line to represent input and response variables
output = forest.predict(test_data)
