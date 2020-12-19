# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:54:34 2020

@author: admin
"""

#Loading Dependencies 
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix


                        #####EXERCISE#####


#Importing Exercise Data Grouping
d1 = pd.read_csv("data_exercise_40.csv", usecols = range(1,120), header = 0)
d1 = d1.fillna(0)
s1 = pd.read_csv("data_exercise_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d1, s1, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Exercise Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Exercise Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Exercise Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Exercise Data Standardized
d10 = pd.read_csv("data_exercise_40_standardized.csv", usecols = range(1,120), header = 0)
d10 = d10.fillna(0)
s10 = pd.read_csv("data_exercise_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d10, s10, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test, predictions)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Exercise Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Exercise Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Exercise Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####FATIGUE#####


#Importing Fatigue Data Grouping
d2 = pd.read_csv("data_fatigue_40.csv", usecols = range(1,120), header = 0)
d2 = d2.fillna(0)
s2 = pd.read_csv("data_fatigue_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d2, s2, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Fatigue Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Fatigue Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Fatigue Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Fatigue Data Standardized
d11 = pd.read_csv("data_fatigue_40_standardized.csv", usecols = range(1,120), header = 0)
d11 = d11.fillna(0)
s11 = pd.read_csv("data_fatigue_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d11, s11, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Fatigue Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Fatigue Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Fatigue Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####PAIN INFERENCE#####


#Importing Pain Inference Data Grouping
d3 = pd.read_csv("data_paininference_40.csv", usecols = range(1,120), header = 0)
d3 = d3.fillna(0)
s3 = pd.read_csv("data_paininference_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d3, s3, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Pain Inference Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Pain Inference Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Pain Inference Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Pain Inference Data Standardized
d12 = pd.read_csv("data_paininference_40_standardized.csv", usecols = range(1,120), header = 0)
d12 = d12.fillna(0)
s12 = pd.read_csv("data_paininference_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d12, s12, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Pain Inference Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Pain Inference Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Pain Inference Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####PHYSICAL FUNCTION#####


#Importing Physical Function Data Grouping
d4 = pd.read_csv("data_physicalfunction_40.csv", usecols = range(1,120), header = 0)
d4 = d4.fillna(0)
s4 = pd.read_csv("data_physicalfunction_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d4, s4, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Physical Function Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Physical Function Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Physical Function Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Physical Function Data Standardized
d13 = pd.read_csv("data_physicalfunction_40_standardized.csv", usecols = range(1,120), header = 0)
d13 = d13.fillna(0)
s13 = pd.read_csv("data_physicalfunction_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d13, s13, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Physical Function Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Physical Function Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Physical Function Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####RA6#####


#Importing RA6 Data 
d5 = pd.read_csv("data_ra6.csv", usecols = range(1,120), header = 0)
d5 = d5.fillna(0)
s5 = pd.read_csv("data_ra6.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d5, s5, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('RA6 Data ')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('RA6 Data kappa score:  ', cohen_kappa_score(y_test, predictions))
print('RA6 Data pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####RA7#####


#Importing RA7 Data 
d6 = pd.read_csv("data_ra7.csv", usecols = range(1,120), header = 0)
d6 = d6.fillna(0)
s6 = pd.read_csv("data_ra7.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d6, s6, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('RA7 Data ')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('RA7 Data kappa score:  ', cohen_kappa_score(y_test, predictions))
print('RA7 Data pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####RA FLARE#####


#Importing RA Flare Data Grouping
d7 = pd.read_csv("data_raflare_40.csv", usecols = range(1,120), header = 0)
d7 = d7.fillna(0)
s7 = pd.read_csv("data_raflare_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d7, s7, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('RA Flare Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('RA Flare Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('RA Flare Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing RA Flare Data Standardized
d16 = pd.read_csv("data_raflare_40_standardized.csv", usecols = range(1,120), header = 0)
d16 = d16.fillna(0)
s16 = pd.read_csv("data_raflare_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d16, s16, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('RA Flare Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('RA Flare Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('RA Flare Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####SLEEP DISTURBANCE#####


#Importing Sleep Disturbance Data Grouping
d8 = pd.read_csv("data_sleepdist_40.csv", usecols = range(1,120), header = 0)
d8 = d8.fillna(0)
s8 = pd.read_csv("data_sleepdist_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d8, s8, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Sleep Disturbance Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Sleep Disturbance Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Sleep Disturbance Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Sleep Disturbance Data Standardized
d17 = pd.read_csv("data_sleepdist_40_standardized.csv", usecols = range(1,120), header = 0)
d17 = d17.fillna(0)
s17 = pd.read_csv("data_sleepdist_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d17, s17, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Sleep Disturbance Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Sleep Disturbance Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Sleep Disturbance Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


                        #####SOCIAL ACTIVITY#####


#Importing Social Activity Data Grouping
d9 = pd.read_csv("data_socialactivity_40.csv", usecols = range(1,120), header = 0)
d9 = d9.fillna(0)
s9 = pd.read_csv("data_socialactivity_40.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d9, s9, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Social Activity Data Grouping')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Social Activity Data Grouping kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Social Activity Data Grouping pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])


#Importing Social Activity Data Standardized
d18 = pd.read_csv("data_socialactivity_40_standardized.csv", usecols = range(1,120), header = 0)
d18 = d18.fillna(0)
s18 = pd.read_csv("data_socialactivity_40_standardized.csv", usecols = range(120,121) , header = 0)

#Split dataset
seed = 0
test_size = .2
X_train, X_test, y_train, y_test = train_test_split(d18, s18, test_size=test_size, random_state=seed)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train.values.ravel())
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in  y_pred.round()]
y_test = np.squeeze(y_test)

#Evaluate 
cm = confusion_matrix(y_test.round(), predictions, normalize = None)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    plt.title('Social Activity Data Standardized')
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)

#print('Social Activity Data Standardized kappa score:  ', cohen_kappa_score(y_test, predictions))
print('Social Activity Data Standardized pearson correlation:   ',  np.corrcoef(y_test, predictions)[0,1])
