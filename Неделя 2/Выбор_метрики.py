# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn import metrics
import sklearn.datasets

X = sklearn.datasets.load_boston().data 
Y = sklearn.datasets.load_boston().target
X_scale = scale(X)
p_values = np.linspace(1.0, 10.0, num=200)
kf = KFold( n_splits=5, shuffle=True, random_state=42)

cv_accuracy = [cross_val_score(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_i, metric='minkowski'), X=X_scale, y=Y, cv=kf).mean() for p_i in p_values]

# Определите, при каком p качество на кросс-валидации оказалось оптимальным (обратите внимание, что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать среднее этих показателей). Это значение параметра и будет ответом на задачу.
best_p = p_values[int(max(cv_accuracy))]
print(best_p)
