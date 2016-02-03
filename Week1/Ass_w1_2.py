# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:38:26 2016

@author: Алексей
"""

from sklearn import tree
import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/titanic.csv',index_col = "PassengerId")

#удаляем лишние признаки
del data['Name']
del data['SibSp']
del data['Parch']
del data['Ticket']
del data['Cabin']
del data['Embarked']

#удаляем записи с пропущенными значениями
data = data[pd.notnull(data['Age'])]

#земеняем текстовые переменные
data.Sex[data.Sex=='male'] = 0
data.Sex[data.Sex=='female'] = 1

#готовим массивы для обучения
Y = np.array(data['Survived'])
del data['Survived']
X = np.array(data)

#обучаем классификатор
clf = tree.DecisionTreeClassifier(random_state = 241)
clf.fit(X, Y)

importances = clf.feature_importances_
