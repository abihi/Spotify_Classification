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

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

dataset = pd.read_csv("training_data.csv")
testset = pd.read_csv("songs_to_classify.csv")

y = dataset.label
X = dataset.drop(['label', 'time_signature'], axis=1)
testset = testset.drop(['time_signature'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0)

#train_labels = tf.concat([1 - y_train, y_train], 0)
#test_labels = tf.concat([1 - y_test, y_test], 0)

train_stats = X.describe()
train_stats = train_stats.transpose()

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)
normed_test_final = norm(testset)

def build_model():
  model = keras.Sequential([
    layers.Dense(10, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(1, activation=tf.nn.sigmoid)
  ])

  optimizer = tf.keras.optimizers.SGD(lr=0.01)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

from sklearn.model_selection import StratifiedKFold, cross_val_score
# Random seed for reproducibility
seed = 7
np.random.seed(seed)

# Define 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
highest_score = 0
for train, test in kfold.split(normed_train_data, y_train):
    # create model
    model = build_model()
    model.summary()

    EPOCHS = 1000

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Fit the model
    history = model.fit(normed_train_data.iloc[train], y_train.iloc[train], epochs=EPOCHS, validation_split = 0.2,
                        verbose=0, callbacks=[early_stop, PrintDot()])

    # Capture model training history
    #hist = pd.DataFrame(history.history)
    #hist['epoch'] = history.epoch
    #print(hist.keys())

    # Evaluate the model
    scores = model.evaluate(normed_train_data.iloc[test], y_train.iloc[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

    #Predict final
    if scores[1] > highest_score:
        final_predictions = model.predict(normed_test_final)
        highest_score = scores[1]

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Loss [target]')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])

#plot_history(history)
#plt.show()

prediction_string = ""
for i in final_predictions:
    if i <= 0.5:
        prediction_string += str(0)
    else:
        prediction_string += str(1)
print(prediction_string)
