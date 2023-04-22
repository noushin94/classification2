
import pandas as pd
import numpy as np

pd.isnull()

pd.dtypes

pd['Diagnosis'].value_counts()

x = pd.drop('Diagnosis', axis=1)
y = pd['Diagnosis']

# using K_fold cross validation  #importing package
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score ,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

GNB = GaussianNB()
pred = cross_val_predict(GNB, x,y, cv=10)
GNB.fit(x,y, sample_weight=None)
print(accuracy_score(y,pred))
print(f1_score(y,pred))
print(confusion_matrix(y,pred))

KNN = KNeighborsClassifier(n_neighbors= 5 , weights= 'uniform', n_jobs=3)
pred = cross_val_predict(KNN, x,y , cv=10)
KNN.fit(x,y)
print(accuracy_score(y,pred))
print(f1_score(y,pred))
print(confusion_matrix(y,pred))

DTC = DecisionTreeClassifier(criterion='entropy' ,max_depth= 4 , min_samples_split= 5 ,min_samples_leaf= 5)
pred = cross_val_predict(DTC, x,y, cv=10)
DTC.fit(x,y, sample_weight= None, check_input= True)
print(accuracy_score(y,pred))
print(f1_score(y,pred))
print(confusion_matrix(y,pred))

MLP =MLPClassifier(hidden_layer_sizes=100, activation='relu' , alpha= 0.001 ,batch_size='auto', learning_rate= 'invscaling', learning_rate_init=0.001)
pred = cross_val_predict(MLP,x,y,cv=100)
MLP.fit(x,y)
print(accuracy_score(y,pred))
print(f1_score(y,pred))
print(confusion_matrix(y,pred))

SVM = SVC(C= 1 , kernel='linear' , class_weight= None)
pred = cross_val_predict(SVM, x,y , cv=10)
SVM.fit(x,y)
print(accuracy_score(y,pred))
print(f1_score(y,pred))
print(confusion_matrix(y,pred))



