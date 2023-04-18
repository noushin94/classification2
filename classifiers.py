import pandas as pd
import numpy as np
df = pd.read_excel(r'DiabetesDiagnosis.xls')
df.head()
df.isnull().sum()
df.dtypes
df['Diagnosis'].value_counts()

X = df.drop('Diagnosis', axis=1)
Y = df['Diagnosis']

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=22)

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(Xtrain, Ytrain)
pred = GNB.predict(Xtest)
pred

from sklearn.metrics import confusion_matrix, accuracy_score
GNB = GaussianNB()
GNB.fit(Xtrain, Ytrain)
pred = GNB.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(Xtrain, Ytrain)
pred = MNB.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, weights='uniform', n_jobs=3)
KNN.fit(Xtrain, Ytrain)
pred = KNN.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=7, min_samples_leaf=2,
                            class_weight={0:0.33 , 1:0.33, 2:0.33}, random_state=2)
DT.fit(Xtrain, Ytrain)
pred = DT.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

pred = DT.predict_proba(Xtest)
pred