# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:23:20 2016

@author: Алексей
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack


#импортируем данные
train = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week4/salary-train.csv')
test = pd.read_csv(
    'C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/Week4/salary-test-mini.csv')
    
#очистка данных
train['FullDescription'] = train['FullDescription'].str.lower()
train['LocationNormalized'] = train['LocationNormalized'].str.lower()
test['FullDescription'] = test['FullDescription'].str.lower()
test['LocationNormalized'] = test['LocationNormalized'].str.lower()

train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
train['LocationNormalized'] = train['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
test['LocationNormalized'] = test['LocationNormalized'].replace('[^a-zA-Z0-9]', ' ', regex = True)

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)     
test['LocationNormalized'].fillna('nan', inplace=True)
test['ContractTime'].fillna('nan', inplace=True)     

vectorizer = TfidfVectorizer(min_df=5)
FullDescription_v = vectorizer.fit_transform(train['FullDescription'])
FullDescription_v_test = vectorizer.transform(test['FullDescription'])

enc = DictVectorizer()
train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))   

#соединяем признаки в матрицу
X_train = hstack([FullDescription_v,train_categ])
Y_train = train['SalaryNormalized']
X_test = hstack([FullDescription_v_test,test_categ])

#обучаем с помощью гребневой регресии
clf = Ridge(alpha=1.0)
clf.fit(X_train, Y_train)

Y_test = clf.predict(X_test)

print(Y_test)

