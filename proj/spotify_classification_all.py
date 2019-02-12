from __future__ import print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import sys
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

def main(model_selection):
    dataset = pd.read_csv("training_data.csv")
    testset = pd.read_csv("songs_to_classify.csv")

    y = dataset.label
    # mode and time_signature are not important features according to xgboost algorithm
    X = dataset.drop(['label', 'mode', 'time_signature'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    #print(X.keys())

    #Basic correlation matrix
    corr = X.corr()
    #plt.matshow(corr)

    #Seaborn correlation matrix (breaks XGB for some reason ??? )
    #mask = np.zeros_like(corr, dtype=np.bool)
    #mask[np.triu_indices_from(mask)] = True
    #f, ax = plt.subplots(figsize=(11, 9))
    #cmap = sns.diverging_palette(220, 10, as_cmap=True)
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    #plt.show()

    #Visualize data (search for correlations)
    #sns.pairplot(X[X.columns.values], diag_kind="kde")
    #plt.show()

    #train_labels = tf.concat([1 - y_train, y_train], 0)
    #test_labels = tf.concat([1 - y_test, y_test], 0)

    train_stats = X.describe()
    train_stats = train_stats.transpose()
    #print train_stats['mean']
    #print train_stats['std']

    def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(X_train)
    normed_test_data = norm(X_test)

    model_used = ""
    history = []

    def build_model_knn():
        global model_used, history
        model_used = "Model K-nearest neighbor"
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=13)
        history = model.fit(normed_train_data, y_train)
        return model

    def build_model_xgboost():
        global model_used, history
        model_used = "Model xgboost (boosting)"
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.5)
        history = xgb_model.fit(normed_train_data, y_train)
        # plot feature importance (least important are mode and time signature)
        #plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
        #plt.show()
        return xgb_model

    def build_model_bagging():
        global model_used, history
        model_used = "Model Bagging"
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier()
        history = model.fit(normed_train_data, y_train)
        return model

    def build_model_random_forest():
        global model_used, history
        model_used = "Model Random forest"
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        history = model.fit(normed_train_data, y_train)
        return model

    def build_model_neural_network():
        global model_used, history
        model_used = "Model neural network"
        model = keras.Sequential([
            layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X_train.keys())]),
            layers.Dense(64, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax)
        ])
        optimizer = tf.keras.optimizers.SGD(lr=0.01)
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
        EPOCHS = 1000
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(normed_train_data, y_train, epochs=EPOCHS, validation_split = 0.2,
                          verbose=0, callbacks=[early_stop, PrintDot()])
        return model

    def build_model_logistic_regression():
        global model_used, history
        model_used = "Model logistic regression"
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(cv=5, random_state=0,
                multi_class='multinomial').fit(normed_train_data, y_train)
        return model

    def build_model_discriminant_analysis():
        global model_used, history
        model_used = "Model discriminant analysis: LDA"
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
        history = model.fit(normed_train_data, y_train)
        return model

    nn = 0
    if  model_selection == "1":
        model = build_model_knn()
    elif model_selection == "2":
        model = build_model_xgboost()
    elif model_selection == "3":
        model = build_model_bagging()
    elif model_selection == "4":
        model = build_model_random_forest()
    elif model_selection == "5":
        model = build_model_neural_network()
        nn = 1
    elif model_selection == "6":
        model = build_model_logistic_regression()
    elif model_selection == "7":
        model = build_model_discriminant_analysis()

    predictions = model.predict(normed_test_data)

    from sklearn.metrics import accuracy_score, classification_report
    if nn == 1:
        preds = [0]*len(predictions)
        for i in range(len(predictions)):
            if predictions[i][1] > 0.5:
                preds[i] = 1
        print(classification_report(y_test, preds))
        print(model_used,"accuracy: %.2f" % (accuracy_score(y_test,preds) * 100),"%")
    else:
        print(classification_report(y_test, predictions))
        print(model_used,"accuracy: %.2f" % (accuracy_score(y_test,predictions) * 100),"%")

    #hist = pd.DataFrame(history.history)
    #hist['epoch'] = history.epoch
    #print(hist.keys())

    #plot_history(history)
    #plt.show()

if sys.argv[1] > 0:
    main(sys.argv[1])
else:
    print("Usage: python spotify_classification_all.py n \n n=1: K-nearest neighbor",
    "\n n=2: XGBoost \n n=3: Bagging \n n=4: Random forest \n n=5: Neural network",
    "\n n=6: Logistic regression \n n=7: Discriminant analysis")
