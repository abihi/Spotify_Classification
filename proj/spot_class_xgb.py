from __future__ import print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("training_data.csv")
testset = pd.read_csv("songs_to_classify.csv")

y = dataset.label
X = dataset.drop(['label', 'time_signature', 'mode', 'key', 'loudness', 'energy'], axis=1)
testset = testset.drop(['time_signature', 'mode', 'key', 'loudness', 'energy'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Visualize data (search for correlations)
#sns.pairplot(X_train[X_train.columns.values], diag_kind="kde")
#plt.show()

train_stats = X.describe()
train_stats = train_stats.transpose()

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)

normed_test_final = norm(testset)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score

def modelfit(alg, dtrain, targets, final_pred, useTrainCV=True, cv_folds=10, early_stopping_rounds=100):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=targets.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], stratified=False,
            nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=None)
        alg.set_params(n_estimators=cvresult.shape[0])
        #print(cvresult.shape[0])

    eval_set = [(dtrain, targets), (normed_test_data, y_test)]

    #Fit the algorithm on the data
    alg.fit(dtrain, targets, eval_metric=['auc', 'error'], eval_set=eval_set, verbose=False)

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    final_predictions = alg.predict(final_pred)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]

    #kfold = KFold(n_splits=10, random_state=7)
    kfold = StratifiedKFold(n_splits=10, random_state=7)
    results = cross_val_score(alg, normed_train_data, y_train, cv=kfold)

    prediction_string = ""
    for i in final_predictions:
        prediction_string += str(i)

    print(prediction_string)

    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(targets, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(targets, dtrain_predprob))
    print ("Accuracy(CV): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.show()

xgb_model = XGBClassifier(
     learning_rate =0.1,
     n_estimators=5000,
     max_depth=6,
     min_child_weight=3,
     gamma=0.2,
     subsample=0.45,
     colsample_bytree=0.35,
     objective= 'binary:logistic',
     nthread=4,
     reg_alpha=0.05,
     scale_pos_weight=1,
     seed=27
    )
modelfit(xgb_model, normed_train_data, y_train, normed_test_final)


#Predictions on test set
#predictions = xgb_model.predict(normed_test_data)
#Predictions on final test set
#preds_final = xgb_model.predict(normed_test_final)
#print(preds_final)

#all_1 = [1]*len(y)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y,all_1))
