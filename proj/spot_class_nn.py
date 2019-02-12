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
X = dataset.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

corr = X.corr()

#Basic correlation matrix
plt.matshow(corr)

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#Visualize data (search for correlations)
sns.pairplot(X[X.columns.values], diag_kind="kde")
plt.show()

train_labels = tf.concat([1 - y_train, y_train], 0)
test_labels = tf.concat([1 - y_test, y_test], 0)

train_stats = X.describe()
train_stats = train_stats.transpose()
#print train_stats['mean']
#print train_stats['std']

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(X_train)
normed_test_data = norm(X_test)

print(len(X_train.keys()))

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(2, activation=tf.nn.softmax)
  ])

  optimizer = tf.keras.optimizers.SGD(lr=0.01)

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

model = build_model()
model.summary()

EPOCHS = 1000

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, y_train, epochs=EPOCHS, validation_split = 0.2,
                    verbose=0, callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.keys())

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

predictions = model.predict(normed_test_data)

preds = [0]*len(predictions)

for i in range(len(predictions)):
    if predictions[i][1] > 0.5:
        preds[i] = 1

from sklearn.metrics import accuracy_score
print("\naccuracy: ", accuracy_score(y_test,preds))
