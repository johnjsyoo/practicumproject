import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Importing raw observations
df=pd.read_csv('/Users/lucaslaviolet/Desktop/Human Activity Recognition/raw_data_v3.csv')

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
    

""" Get Features for Windows of Data from All Sensors """
data = window_stats(df, 500)
labels = data.ix[:,0]
all_features = data.ix[:,2:].astype(float)


""" Standardize the Range of Features """
features_scaled = pd.DataFrame(preprocessing.scale(all_features), columns=all_features.columns)
#means = features_scaled.mean(axis=0)

""" Specify Only Features from Sensors of Interest """
# May want to find a way to easily exclude temp columns

ankle = features_scaled[[c for c in features_scaled.columns if "ankle" in c]]
chest = features_scaled[[c for c in features_scaled.columns if "chest" in c]]
hand = features_scaled[[c for c in features_scaled.columns if "hand" in c]]

#hr = data[[c for c in data.columns if "heart" in c]]
#temp = data[[c for c in data.columns if "temp" in c]]
#
#ankle_hr = data[[c for c in data.columns if any(word in c for word in ['ankle','heart'])]]
#chest_hr = data[[c for c in data.columns if any(word in c for word in ['chest','heart'])]]
#hand_hr = data[[c for c in data.columns if any(word in c for word in ['hand','heart'])]]
#
#ankle_chest = data[[c for c in data.columns if any(word in c for word in ['ankle','chest'])]]
#ankle_hand = data[[c for c in data.columns if any(word in c for word in ['ankle','hand'])]]
#chest_hand = data[[c for c in data.columns if any(word in c for word in ['chest','hand'])]]
#
#ankle_chest_hand = data[[c for c in data.columns if any(word in c for word in ['ankle','chest','hand'])]]


"""COMPARE MODEL RESULTS FOR DIFFERENT SENSORS"""
# Output a dictionary with 3 key:value pairs: 
#   1) Classification reports for each model
#   2) Confusion matrices for each model
#   3) Scores for each model
def model(features, labels):
    """ Specify Features and Labels """
    X = features
    Y = labels
    
    """ Cross Validation """
    
    from sklearn.cross_validation import StratifiedKFold
    n_folds=5
    skf = StratifiedKFold(Y, n_folds)
     
    """ Modeling """
    # Name and fit models 
    svm_05 = LinearSVC(C=0.5)
    svm_1 = LinearSVC(C=1.0)
    svm_2 = LinearSVC(C=2.0)
    
    # Print classification report and average of fold scores
    scores = []
    for name, clf in [('Support Vector Machine (C = 0.5)', svm_05),
                      ('Support Vector Machine (C = 1.0)', svm_1),
                      ('Support Vector Machine (C = 2.0)', svm_2)]:
        clf_score = sum([clf.fit(X.iloc[train], Y.iloc[train]).score(X.iloc[test],\
        Y.iloc[test]) for train, test in skf])/n_folds
        print(name, ': \n', classification_report(Y, clf.predict(X)))
        scores.append((name, clf_score))
    
#    # Report confusion matrix for each model
#    knn_2_conf = confusion_matrix(Y, knn_2.predict(X))
#    knn_3_conf = confusion_matrix(Y, knn_3.predict(X))
#    knn_4_conf = confusion_matrix(Y, knn_4.predict(X))
#    knn_5_conf = confusion_matrix(Y, knn_5.predict(X))
#    knn_6_conf = confusion_matrix(Y, knn_6.predict(X))
#    knn_7_conf = confusion_matrix(Y, knn_7.predict(X))
#    knn_8_conf = confusion_matrix(Y, knn_8.predict(X))
#    knn_9_conf = confusion_matrix(Y, knn_9.predict(X))
#    knn_10_conf = confusion_matrix(Y, knn_10.predict(X))
#    
#    conf_matrices = [knn_2_conf,knn_3_conf,knn_4_conf,knn_5_conf,knn_6_conf,\
#    knn_7_conf,knn_8_conf,knn_9_conf,knn_10_conf]
#    
#    # Report classification report for each model
#    knn_2_class = classification_report(Y, knn_2.predict(X))
#    knn_3_class = classification_report(Y, knn_3.predict(X))
#    knn_4_class = classification_report(Y, knn_4.predict(X))
#    knn_5_class = classification_report(Y, knn_5.predict(X))
#    knn_6_class = classification_report(Y, knn_6.predict(X))
#    knn_7_class = classification_report(Y, knn_7.predict(X))
#    knn_8_class = classification_report(Y, knn_8.predict(X))
#    knn_9_class = classification_report(Y, knn_9.predict(X))
#    knn_10_class = classification_report(Y, knn_10.predict(X))
#    
#    class_reports = [knn_2_class,knn_3_class,knn_4_class,knn_5_class,\
#    knn_6_class,knn_7_class,knn_8_class,knn_9_class,knn_10_class]
#    
#    return {'scores':scores,'conf_matrices':conf_matrices,'class_reports':class_reports}
    return scores

ankle_results = model(ankle, labels)
chest_results = model(chest, labels)
hand_results = model(hand, labels)

#hr_results = model(hr, labels)
#temp_results = model(temp, labels)
#
#ankle_chest_results = model(ankle_chest, labels)
#ankle_hand_results = model(ankle_hand, labels)
#chest_hand_results = model(chest_hand, labels)
#
#ankle_chest_hand_results = model(ankle_chest_hand,labels)
#
#all_features_results = model(all_features, labels)