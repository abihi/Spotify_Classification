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
X = dataset.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)

#Visualize data (search for correlations)
#sns.pairplot(X_train[X_train.columns.values], diag_kind="kde")
#plt.show()

train_stats = X.describe()
train_stats = train_stats.transpose()
#print train_stats['mean']
#print train_stats['std']

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)

normed_test_final = norm(testset)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Performing grid search

def modelfit(alg, dtrain, targets, final_pred, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain.values, label=targets.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], stratified=False,
            nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=None)
        alg.set_params(n_estimators=cvresult.shape[0])
        #print(cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain, targets, eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    final_predictions = alg.predict(final_pred)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]

    prediction_string = ""
    for i in final_predictions:
        prediction_string += str(i)

    print(prediction_string)

    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(targets, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(targets, dtrain_predprob))
    #https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.show()

xgb_model = XGBClassifier(
     learning_rate =0.1,
     n_estimators=1000,
     max_depth=6,
     min_child_weight=3,
     gamma=0.2,
     subsample=0.45,
     colsample_bytree=0.35,
     objective= 'binary:hinge',
     nthread=4,
     reg_alpha=0.05,
     scale_pos_weight=1,
     seed=27
    )
modelfit(xgb_model, normed_train_data, y_train, normed_test_final)


from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#kfold = KFold(n_splits=10, random_state=7)
#kfold = StratifiedKFold(n_splits=10, random_state=7)
#results = cross_val_score(xgb_model, normed_train_data, y_train, cv=kfold)
#print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#Predictions on test set
#predictions = xgb_model.predict(normed_test_data)
#Predictions on final test set
#preds_final = xgb_model.predict(normed_test_final)
#print(preds_final)

#all_1 = [1]*len(y)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y,all_1))
