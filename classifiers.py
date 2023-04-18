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