# -*- coding: utf-8 -*-
"""Введение в машинное обучение неделя 1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B8jV9OKxUoTEMjiDGcRk7saon2Ndo9WT
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
col_list = ["Pclass", "Age","Fare","Sex","Survived"]
file = pd.read_csv(f'./drive/My Drive/titanic.csv', usecols=col_list)
file.dropna(subset = ["Pclass", "Age","Fare","Sex"], inplace=True)
file["Sex"] = file["Sex"].replace(['male'],0)
file["Sex"] = file["Sex"].replace(['female'],1)
data = ['Sex','Age','Fare','Pclass']
X = file[data]
y = file.Survived
clf = DecisionTreeClassifier(random_state=123)
clf.fit(X, y)

feat_importance = clf.tree_.compute_feature_importances(normalize=False)
for i in range(len(feat_importance)):
  print("%s = "%data[i],feat_importance[i])
