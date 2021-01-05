# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

 


newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target
print("Скачал данные")


vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)
print("Вычислил TF-IDF-признаки")


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(model, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)
best_C = gs.best_params_['C']
print("Вычислил С")


model = SVC(kernel='linear', random_state=241, C=best_C)
model.fit(vectorizer.transform(X), y)

print("Обучил SVC")
words = vectorizer.get_feature_names()
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
print(top_words)
