# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:33:45 2020

@author: Al Rahrooh
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


#Importing DATA
data1 = pd.read_csv("data_exercise_40.csv", usecols = range(1,120), header = 0)
data1 = data1.fillna(0)
s1 = pd.read_csv("data_exercise_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(data1, s1, test_size=test_size, random_state=seed)

#fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
#make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, predictions))


#regression test
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10, verbosity = 0)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

data_dmatrix = xgb.DMatrix(data1,s1)

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
print((cv_results["test-rmse-mean"]).tail(1))


#Confusion Matrix
cm = confusion_matrix(y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)