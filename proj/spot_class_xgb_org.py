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
X = dataset.drop(['label', 'time_signature'], axis=1)
testset = testset.drop(['time_signature'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#Visualize data (search for correlations)
#sns.pairplot(X_train[X_train.columns.values], diag_kind="kde")
#plt.show()

train_stats = X.describe()
train_stats = train_stats.transpose()
#print train_stats['mean']
#print train_stats['std']

def norm(x):
  return (x - train_stats['mean']) / (train_stats['std']*train_stats['std'])

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)
normed_test_final = norm(testset)

#Change model here to your own
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    learning_rate =0.01,
    n_estimators=5000,
    max_depth=10,
    min_child_weight=1,
    gamma=0.06,
    subsample=0.7,
    colsample_bytree=0.5,
    objective= 'binary:logistic',
    nthread=1,
    reg_alpha=0.05,
    scale_pos_weight=2,
    seed=27
)
history = xgb_model.fit(normed_train_data, y_train)

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(xgb_model, normed_train_data, y_train, cv=kfold)
print("Accuracy (CV): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#Predictions on final test set
final_predictions = xgb_model.predict(normed_test_final)
prediction_string = ""
for i in final_predictions:
    prediction_string += str(i)
print(prediction_string)

#Predictions on test set
predictions = xgb_model.predict(normed_test_data)
from sklearn.metrics import accuracy_score
print("Accuracy (test set): %.2f%%" % accuracy_score(y_test,predictions))
