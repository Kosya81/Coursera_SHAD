# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:41:54 2016

@author: Алексей
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week2/perceptron-train.csv',
    header=None)
data_test = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week2/perceptron-test.csv',
    header=None)

#готовим массивы для обучения
y_train = np.array(data_train[0])
del data_train[0]
x_train = np.array(data_train)

#готовим массивы для тестов
y_test = np.array(data_test[0])
del data_test[0]
x_test = np.array(data_test)

#обучаем перцептрон
clf = Perceptron(random_state=241)
clf.fit(x_train,y_train)

#делаем предсказания по тестовой выборке
y_test_pred = clf.predict(x_test)

#оценим точность на тестовой выборке
score_test = accuracy_score(y_test, y_test_pred)

#нормализуем выборки
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf.fit(x_train_scaled,y_train)
y_test_pred_scaled = clf.predict(x_test_scaled)
score_test_scaled = accuracy_score(y_test, y_test_pred_scaled)

#получаем ответ на задание
ans = score_test_scaled - score_test