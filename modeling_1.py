import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

df=pd.read_csv('raw_data_v3.csv')

""" Windows and Feature Selection """

#Dropping transient activity
df = df[df['activity_id'] != 0]

# Sorting by subject and timestamp
df = df.sort(['subject_id','timestamp'])

test = df[:10000]

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
            #print(window.dtypes)            
    d = pd.concat(all_stats,axis=1).T
    d.columns = ['activity','subject_id'] + [x + '_mean' for x in df.columns[3:34]] + \
    [x + '_std' for x in df.columns[3:34]] + [x + '_range' for x in df.columns[3:34]] 
    return d    
    
wd_data = window_stats(df, 500)

#wd_data.ix[:,3:96] = wd_data.ix[:,3:96].astype(float)

""" Setting the Indices for K-Fold Cross Validation """
X = wd_data.ix[:,3:96].astype(float)
Y = wd_data.ix[:,0]

""" Modeling - K-Nearest Neighbors """

#wd_data.ix[:3,:3]

#wd_data.dtypes
#X.dtypes

#wd_data_scaled = pd.concat(wd_data.ix[:,:2],preprocessing.normalize(wd_data.ix[:,3:]))

#rwar = preprocessing.normalize(wd_data.ix[:,3:96])

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y) 

print(neigh.predict(X))

output = neigh.predict(X)
neigh.score(X,Y)

""" Cross-Validating """

#regular CV

from sklearn import cross_validation

labels = Y

kfold = cross_validation.KFold(len(X), n_folds=5)
[neigh.fit(X.iloc[train], Y.iloc[train]).score(X.iloc[test], Y.iloc[test]) for train, test in kfold]

#strat CV

from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(labels, 5)

d = [neigh.fit(X.iloc[train], Y.iloc[train]).score(X.iloc[test], Y.iloc[test]) for train, test in skf]
sum(d)/5



#print(neigh.predict_proba([[0.9]]))


""" Logistic Regression """
from sklearn.linear_model import LogisticRegression

kfold = cross_validation.KFold(len(wd_data), n_folds=10, shuffle=True, random_state=4)

lr = LogisticRegression()

cross_validation.cross_val_score(lr, X, Y, cv=kfold, n_jobs = 1)


""" Random Forests """
from sklearn.ensemble import RandomForestClassifier

kfold = cross_validation.KFold(len(wd_data), n_folds=10, shuffle=True, random_state=4)

forest = RandomForestClassifier(n_estimators = 100)       # Tuning Parameters, see Documentation
forest = forest.fit(X_train, y_train)                     # Fix this line to represent input and response variables
output = forest.predict(X_test)

cross_validation.cross_val_score(forest, X, Y, cv=kfold, n_jobs = 1)
