# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:50:21 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn import metrics as m


cls = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week3/classification.csv')

TP = len(cls[(cls.true == 1) & (cls.pred == 1)])
FP = len(cls[(cls.true == 0) & (cls.pred == 1)])
FN = len(cls[(cls.true == 1) & (cls.pred == 0)])
TN = len(cls[(cls.true == 0) & (cls.pred == 0)])

print('TP: %i, FP: %i, FN: %i, TN: %i' %(TP,FP,FN,TN))

Accuracy = m.accuracy_score(cls.true,cls.pred)
Precision = m.precision_score(cls.true,cls.pred)
Recall = m.recall_score(cls.true,cls.pred)
F1 = m.f1_score(cls.true,cls.pred)

print('Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1: %.2f'
    %(Accuracy,Precision,Recall,F1))

scores = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week3/scores.csv')

score_logreg = m.roc_auc_score(scores.true,scores.score_logreg)
score_svm = m.roc_auc_score(scores.true,scores.score_svm)
score_knn = m.roc_auc_score(scores.true,scores.score_knn)
score_tree = m.roc_auc_score(scores.true,scores.score_tree)

print('logreg: %.3f, svm: %.3f, knn: %.3f, tree: %.3f'
    %(score_logreg,score_svm,score_knn,score_tree))
    
logreg = m.precision_recall_curve(scores.true,scores.score_logreg)
print('Maximum precision on logreg: %.3f' %max(logreg[0][logreg[1]>=0.7]))

svm = m.precision_recall_curve(scores.true,scores.score_svm)
print('Maximum precision on svm: %.3f' %max(svm[0][svm[1]>=0.7]))

knn = m.precision_recall_curve(scores.true,scores.score_knn)
print('Maximum precision on knn: %.3f' %max(knn[0][knn[1]>=0.7]))

tree = m.precision_recall_curve(scores.true,scores.score_tree)
print('Maximum precision on tree: %.3f' %max(tree[0][tree[1]>=0.7]))
