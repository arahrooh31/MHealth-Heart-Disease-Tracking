# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:17:15 2020

@author: Al Rahrooh
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report,f1_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import KFold,GridSearchCV,RandomizedSearchCV,GroupShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns



path='data_paininference_40_standardized.csv'#path of the data file
df=pd.read_csv(path,header=0)
s='_1_Weekly_PainInterference_Score'
df=df[df['_1_Weekly_PainInterference_Score'].notna()]
df.loc[:,"DailyTotalSleepMins1":"DailySedentaryAct7"]=df.loc[:,"DailyTotalSleepMins1":"DailySedentaryAct7"].fillna(
    df.loc[:,"DailyTotalSleepMins1":"DailySedentaryAct7"].mean())# fill empty features by its mean
data=pd.DataFrame(df).values
#data.head()

s1=df.loc[:,[s,'ID']]
uscore=sorted(list(set(s1[s])))
n_class=len(uscore)#number of states(unique score)

feature=data[:,:-3]
label=data[:,-1].astype(int)
groups=data[:,-2]


# random forest model

feature_importance=np.zeros((1,feature.shape[1]))
n=5
tauc=np.zeros([n,1])
prauc=np.zeros([n,1])
kappas=np.zeros([n,1])
rfpred_all=np.array([],dtype=int)# to store all the predictions by rf
rflabels=np.array([],dtype=int)# to store all the labels
 
for i in range(n):
    gss=GroupShuffleSplit(n_splits=10,test_size=0.1,random_state=i)
    
    tlabel=np.array([],dtype=int)
    tprediction=np.array([],dtype=int)
    tscore=np.zeros([1,n_class])
    
    for train_index, test_index in gss.split(feature, label, groups=groups):
        trainfeature, testfeature = feature[train_index], feature[test_index]
        trainlabel, testlabel = label[train_index], label[test_index]    
        rf=RandomForestClassifier(n_estimators=4000,criterion='entropy',max_features=1,
                                  max_depth=None,min_samples_split=150,n_jobs=-1)# hyperparameter for all feature
        rf.fit(trainfeature,trainlabel)
        predictions=rf.predict(testfeature)
        score=rf.predict_proba(testfeature) 
        importances = rf.feature_importances_
        feature_importance += importances
        tlabel=np.concatenate([tlabel,testlabel])
        tprediction=np.concatenate([tprediction,predictions])
        tscore=np.concatenate([tscore,score])
        
    tscore=tscore[1:,:]# delete the first row
    kappas[i]=cohen_kappa_score(tlabel, tprediction)
    
    
print(kappas)

print( "kappa score:  ", cohen_kappa_score(tlabel, tprediction))

cm = confusion_matrix(tlabel, tprediction)

def plot_confusion_matrix(cm, normalized=True, cmap='bone'):
    plt.figure(figsize=[7, 6])
    norm_cm = cm
    if normalized:
        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(norm_cm, annot=cm, fmt='g')

plot_confusion_matrix(cm)