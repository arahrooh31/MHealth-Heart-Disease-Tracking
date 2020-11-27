# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 00:28:56 2020

@author: admin
"""
#Loading Dependencies 
import pandas as pd
import numpy as np
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GridSearchCV


#Importing DATA
data1 = pd.read_csv("data_paininference_40.csv", usecols = range(1,120), header = 0)
#data1 = data1.fillna(0)
s1 = pd.read_csv("data_paininference_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(data1, s1, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, predictions.round()))

#confusion matrix and kappa score for testing set

cm = confusion_matrix(y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

print( "kappa score:  ", cohen_kappa_score(y_test, predictions.round()))


#confusion matrix and kappa score for training set

y_train_pred = model.predict(X_train)
predictions1 = [round(value) for value in y_train_pred]


cm1 = confusion_matrix(y_train, predictions1)

def plot_confusion_matrix(cm1, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm1 = cm1
    if normalized:
        norm_cm1 = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm1, annot=cm1, fmt='g')

plot_confusion_matrix(cm1)

print( "kappa score:  ", cohen_kappa_score(y_train, predictions1))
