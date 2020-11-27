# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:29:12 2020

@author: Al Rahrooh
"""

#Loading Dependencies 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,cohen_kappa_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)


data1 = pd.read_csv('data_exercise_40_standardized.csv', usecols = range(1,120), header = 0)
data1.fillna(data1.mean(), inplace = True)
s1 = pd.read_csv('data_exercise_40_standardized.csv', usecols = range(120,121) , header = 0)

X_train, X_test, y_train, y_test = train_test_split(data1, s1, test_size = .3, random_state = 109)


from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

#Create SVM Classifier
clf = svm.SVC(kernel = 'linear')
#Train the model using the training sets
clf.fit(X_train, y_train.values.ravel())
#Predict the response for the test dataset
y_pred = clf.predict(X_test)


#Confusion matrix and kappa score for testing set

predictions = [round(value) for value in y_pred]

cm = confusion_matrix(y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

print( "kappa score:  ", cohen_kappa_score(y_test, predictions))