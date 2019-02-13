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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

#Change model here to your own
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    learning_rate =0.01,
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
history = xgb_model.fit(normed_train_data, y_train)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(xgb_model, normed_train_data, y_train, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#Predictions on test set
#predictions = xgb_model.predict(normed_test_data)
#Predictions on final test set
#preds_final = xgb_model.predict(normed_test_final)
#print(preds_final)

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test,predictions))
