# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 02:15:46 2020

@author: admin
"""
#Loading Dependencies 
import pandas as pd
import numpy as np
from numpy import loadtxt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

#Importing DATA
data1 = pd.read_csv("PROsMLmodels1/Data_40Grouping/data_exercise_40.csv", usecols = range(1,120), header = 0)
data1.fillna(data1.mean(), inplace = True)
s1 = pd.read_csv('PROsMLmodels1/Data_40Grouping/data_exercise_40.csv', usecols = range(120,121) , header = 0)

#Split dataset
seed = 7
test_size = .33
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
cv_results.head()

print((cv_results["test-rmse-mean"]).tail(1))



xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()



start_time=time.time()


model = XGBRegressor()

param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'min_child_weight': np.arange(0.0001, 0.5, 0.001),
        'gamma': np.arange(0.0,40.0,0.005),
        'learning_rate': np.arange(0.0005,0.3,0.0005),
        'subsample': np.arange(0.01,1.0,0.01),
        'colsample_bylevel': np.round(np.arange(0.1,1.0,0.01)),
        'colsample_bytree': np.round(np.arange(0.1,1.0,0.01))}

kfold = KFold(n_splits=10, shuffle=True, random_state=10)
grid_search = RandomizedSearchCV(model, param_grid, scoring="accuracy", n_iter = 500, cv=kfold)
grid_result = grid_search.fit(data1, s1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]

print(time.time()-start_time)






