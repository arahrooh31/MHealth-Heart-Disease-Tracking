# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:56:48 2020

@author: Al Rahrooh
"""

#Importing Dependencies
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#Importing DATA
data1 = pd.read_csv("PROsMLmodels1/Data_40Grouping/data_exercise_40.csv", usecols = range(1,120), header = 0)
data1.fillna(data1.mean(), inplace = True)
s1 = pd.read_csv('PROsMLmodels1/Data_40Grouping/data_exercise_40.csv', usecols = range(120,121) , header = 0)

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(data1, s1)

#Fit Estimator
est = GradientBoostingClassifier(n_estimators = 300, max_depth = 5)
est.fit(X_train, y_train.values.ravel())

#Predict Class Labels
pred = est.predict(X_test)

#Score on test data
acc = est.score(X_test, y_test)
print('ACC: %.4f' % acc)

#Predict class probabilities
est.predict_proba(X_test)[0]

#HyperParameter Tuning
param_grid = {'learning_rate': [0.1, 0.01, 0.001],
              'max_depth': [3, 10],
              'min_samples_leaf': [3, 5, 7, 10],  
              'max_features': [1.0, 0.3, 0.1] 
              }

est = GradientBoostingRegressor(n_estimators=3000)
gs_cv = GridSearchCV(est, param_grid).fit(X_train, y_train.values.ravel())
print('Best hyperparameters: %r' % gs_cv.best_params_)

#Refit model on best parameters
est.set_params(**gs_cv.best_params_)
est.fit(X_train, y_train.values.ravel())



#Tuning
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train.values.ravel())
print("accuracy on training: %f" % gbrt.score(X_train,y_train.values.ravel()))
print('\n'"accuracy on test: %f" % gbrt.score(X_test,y_test))

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train.values.ravel())
print('\n'"accuracy on training set: %f" % gbrt.score(X_train, y_train.values.ravel()))
print('\n'"accuracy on test set: %f" % gbrt.score(X_test, y_test))

 
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train.values.ravel())
print('\n'"accuracy on training set: %f" % gbrt.score(X_train, y_train.values.ravel()))
print('\n'"accuracy on test set: %f" % gbrt.score(X_test, y_test))

#Regression Test
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train.values.ravel())
GradientBoostingRegressor(random_state=0)
reg.predict(X_test[1:2])
reg.score(X_test, y_test)

#Training GBR Model
params = {'n_estimators': 3, 'max_depth': 3, 'learning_rate': 1, 'criterion': 'mse'}
gradient_boosting_regressor_model = GradientBoostingRegressor(**params)
gradient_boosting_regressor_model.fit(data1, s1.values.ravel())

#Evaluating the Model (x,y need to be same size to run)
#plt.figure(figsize = (12,6))
#plt.scatter(data1, s1.values.ravel())
#plt.plot(data1, gradient_boosting_regressor_model.predict(data1), color = 'black')
#plt.show()

#Evaluate the model
y_pred = est.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Precision:', metrics.precision_score(y_test, y_pred, average = None))
print('Recall:', metrics.recall_score(y_test, y_pred, average = None))
