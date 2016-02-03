# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/Алексей/Documents/Python Scripts/Coursera_SHAD/titanic.csv',index_col = "PassengerId")

Ans1 = data['Sex'].value_counts()

Ans2 = np.mean(data['Survived'])*100

Ans3 = 100*data.Pclass[data.Pclass==1].count()/data.Pclass.count()

Ans4 =(np.mean(data.Age),np.median(data.Age[data.Age>0]))

data_corr = pd.DataFrame()
data_corr['SibSp'] = data['SibSp']
data_corr['Parch'] = data['Parch']

Ans5 = data_corr.corr(method='pearson')

names = pd.DataFrame()
names['Name'] = data.Name[data.Sex == 'female']
n_arr = []

pattern = r'Miss.'
for name in names['Name']:
    if 'Miss.' in name:
        tmp = name.split('Miss.')
        tmp = tmp[1].split()
        n_arr.append(tmp[0])
    elif 'Mrs.' in name:
        tmp = name.split('(')
        try:
            tmp = tmp[1].split()
            n_arr.append(tmp[0])
        except:
            pass

names2 = pd.DataFrame()
names2['Name'] = n_arr
Ans6 = names2['Name'].value_counts()
