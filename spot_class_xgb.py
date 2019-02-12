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

dataset = pd.read_csv("data.csv")

y = dataset.target
artists = dataset.artist
song_titles = dataset.song_title
X = dataset.drop(['target', 'Unnamed: 0', 'song_title', 'artist'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Visualize data (search for correlations)
#sns.pairplot(X_train[X_train.columns.values], diag_kind="kde")
#plt.show()

train_stats = X_train.describe()
train_stats = train_stats.transpose()
#print train_stats['mean']
#print train_stats['std']

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)

from xgboost import XGBRegressor
xgb_model = XGBRegressor()
xgb_model.fit(normed_train_data, y_train)

predictions = xgb_model.predict(normed_test_data)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions.round()))
