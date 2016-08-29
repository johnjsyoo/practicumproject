import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import StratifiedShuffleSplit


# Importing raw observations
df=pd.read_csv('/Users/lucaslaviolet/Desktop/Human Activity Recognition/raw_data_v3.csv')

""" Windows and Feature Creation """

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
            #Statistical Features            
            means = window.ix[:,3:34].mean()
            stds = window.ix[:,3:34].std()
            ranges = window.ix[:,3:34].apply(lambda x: x.max() - x.min())
            kurtos = window.ix[:,3:34].kurtosis()
            skews = window.ix[:,3:34].skew()

            #Structural Features  
            X = pd.Series(range(0,len(window))).reshape(len(window),1)
            Y = window.ix[:,3]   
            lm = LinearRegression().fit(X,Y)
            m_heart = pd.Series(lm.coef_)            
            c_heart = pd.Series(lm.intercept_)            

            window_stats = pd.concat([pd.Series(window.iloc[0]['activity']),\
            pd.Series(window.iloc[0]['subject_id']),means.astype(float),\
            stds.astype(float),ranges.astype(float),kurtos.astype(float),\
            skews.astype(float),m_heart,c_heart],axis=0)
            all_stats.append(window_stats)
    d = pd.concat(all_stats,axis=1).T
    d.columns = ['activity','subject_id'] + [x + '_mean' for x in df.columns[3:34]] + \
    [x + '_std' for x in df.columns[3:34]] + [x + '_range' for x in df.columns[3:34]] + \
    [x + '_kurt' for x in df.columns[3:34]] + [x + '_skew' for x in df.columns[3:34]] + \
    ['m_heart'] + ['c_heart']
    return d

#Get features for windows of data from all sensors
data = window_stats(df, 2500)

""" TEST PREDICTIONS """

#Specify the activity labels (alphabetical list) and features (columns) of interest
activities = sorted(data['activity'].unique())
features = [c for c in data.columns if any(word in c for word in ['chest','heart'])]

""" Split into Training (All But One Subject) and Test (One Subject) Sets """
#Combine 1.0 and 1.1 into the same subject id
data['subject_id'] = np.around(data['subject_id'].astype(np.double),0)

subject_scores = []
subject_confs = []
for subject in range(1,10):
    train = data[data['subject_id'] != subject]
    test = data[data['subject_id'] == subject]

    X_train, X_test = train.ix[:,features], test.ix[:,features]
#    X_train, X_test = train.ix[:,2:], test.ix[:,2:] #Train on all columns instead
    y_train, y_test = train.ix[:,0], test.ix[:,0]

    clf = RandomForestClassifier(n_estimators=300, max_features=0.1).fit(X_train, y_train)
    subject_scores.append([subject, clf.score(X_test, y_test)])
    print(subject, ": ", clf.score(X_test, y_test))
    conf = confusion_matrix(y_test, clf.predict(X_test), labels=activities)
    subject_confs.append(pd.DataFrame(conf, index=activities, columns=activities))

""" Split into Training and Test Sets Using Stratified Shuffle Split """
labels = data.ix[:,0]

split_scores = []
split_confs = []
for test_size in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
    sss = StratifiedShuffleSplit(labels, n_iter=1, test_size=test_size, random_state=0)
    for train_index, test_index in sss:
        X_train = data[features].iloc[list(train_index)]
        X_test = data[features].iloc[list(test_index)]
        y_train = labels.iloc[list(train_index)]
        y_test = labels.iloc[list(test_index)]
    clf = RandomForestClassifier(n_estimators=300, max_features=0.1).fit(X_train, y_train)
    split_scores.append([test_size, clf.score(X_test, y_test)])
    print("test fraction", test_size, ": ", clf.score(X_test, y_test))
    conf = confusion_matrix(y_test, clf.predict(X_test), labels=activities)
    split_confs.append(pd.DataFrame(conf, index=activities, columns=activities))

""" Create Matrix of Counts by Activity and Subject """
comparison = pd.crosstab(data['activity'], data['subject_id'])
comparison_test = pd.crosstab(y_test['activity'], y_test['subject_id'])


""" Rerun Subject 4 Predictions and Save Predicted and Actual Activities """
train = data[data['subject_id'] != 4]
test = data[data['subject_id'] == 4]

X_train, X_test = train.ix[:,features], test.ix[:,features]
y_train, y_test = train.ix[:,0], test.ix[:,0]

clf = RandomForestClassifier(n_estimators=300, max_features=0.1).fit(X_train, y_train)
subject_scores.append([subject, clf.score(X_test, y_test)])
print(subject, ": ", clf.score(X_test, y_test))
predictions = pd.concat([y_test.reset_index()['activity'], \
pd.Series(clf.predict(X_test)).reset_index()[0]], axis=1)
predictions.columns = ['actual','predicted']
predictions.to_csv('/Users/lucaslaviolet/Desktop/Human Activity Recognition/subject_4_predictions.csv')