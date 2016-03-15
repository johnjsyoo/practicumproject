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
for name, clf in [('Logistic', LogisticRegression()),
                  ('Naive Bayes', GaussianNB()),
                  ('Support Vector Classification', LinearSVC(C=1.0)),
                  ('Random Forest', RandomForestClassifier(n_estimators=100)),
                  ('K Nearest Neighbors (N = 3)', KNeighborsClassifier(n_neighbors=3)),
                  ('Bagging', BaggingClassifier(n_estimators=100)),
                  ('Ada Boosting', AdaBoostClassifier(n_estimators=100)),
                  ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100,
                                                                    learning_rate=1.0,
                                                                    max_depth=1,
                                                                    random_state=4))]:
    clf_score = sum([clf.fit(X.iloc[train], Y.iloc[train]).score(X.iloc[test],\
    Y.iloc[test]) for train, test in skf])/n_folds
    print(name, ': \n', classification_report(Y, clf.predict(X)))
    scores.append((name, clf_score))

