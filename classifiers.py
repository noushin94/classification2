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

from sklearn.tree import export_graphviz
export_graphviz(DT, out_file="tree.dot",class_names=True,feature_names=X.columns, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(max_iter=200, random_state=2020, activation='relu', hidden_layer_sizes=(10,20),
                    learning_rate_init=0.01, learning_rate='invscaling')

MLP.fit(Xtrain, Ytrain)
pred = MLP.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

from sklearn.svm import SVC
SV_model = SVC(kernel='linear', gamma=0.01, degree=3, C=0.1, class_weight='balanced')
SV_model.fit(Xtrain, Ytrain)
pred = SV_model.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced', fit_intercept=True)
LR.fit(Xtrain, Ytrain)
pred = LR.predict(Xtest)

print(confusion_matrix(Ytest, pred))
print(accuracy_score(Ytest, pred))

