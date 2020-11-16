# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
from sklearn.metrics import classification_report

#Importing DATA
data = pd.read_csv('data_exercise_40.csv')

#PreProcessing
data1 = pd.read_csv("data_exercise_40.csv", usecols = range(1,120), header = 0)
data1.fillna(data1.mean(), inplace = True)
s1 = pd.read_csv('data_exercise_40.csv', usecols = range(120,121) , header = 0)

#Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(data1, s1, test_size = .3, random_state = 109)


#Create SVM Classifier
clf = svm.SVC(kernel = 'linear')
#Train the model using the training sets  (long runtime)
clf.fit(X_train, y_train.values.ravel())
#Predict the response for the test dataset
y_pred = clf.predict(X_test)
print(y_pred)

#Evaluate the model
print(classification_report(y_test, y_pred))

#Model Accuracy
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
#Model Precision
print('Precision:', metrics.precision_score(y_test, y_pred))
#Model Recall
print('Recall:', metrics.recall_score(y_test, y_pred))

#Tuning Hyperparamters
X_train, X_test, y_train, y_test = train_test_split(data1, s1, test_size = .3, random_state = 32)

kernel = ['linear', 'rbf', 'poly', 'sigmoid']
for i in kernel:
    model = SVC(kernel = i, C = 1.0)
    model.fit(X_train, y_train.values.ravel())
    print('For kernel:', i)
    print('accuracy is:', model.score(X_test, y_test))

model = SVC()
model.fit(X_train, y_train)
print('Accuracy on testing data is:', model.score(X_test, y_test))
print('Accuracy on training data is:', model.score(X_train, y_train))

for i in range(1,10):
    model = SVC(kernel = 'poly', degree = i, C=100)
    model.fit(X_train, y_train)
    print('Accuracy on testing data is: \t', model.score(X_test, y_test))
    print('Accuracy on training data is: \t', model.score(X_train, y_train))
    
param_grid = {'C':[0.1,1,100,1000], 'kernel':['rbf', 'poly', 'sigmoid', 'linear'], 'degree':[1,2,3,4,5,6]}
grid = GridSearchCV(SVC(), param_grid)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.score(X_test, y_test))