# classification2
#use different classifiers on specific datas
##this project is about different classifiers and evaluating their results with specific data(Diabets Diagnosis)
### first I imported pandas and numpy
### then the data can be read by one of the tool of pandas
### in line 5 I estimated whether the data has missing values or not
### in line 6 I captured features values types
### in line 7 with value_counts we can figure out how many people have or have not diabetes
### in line 8 with  model_selection in  package of scikit learn we can split test and train datas
## GaussianNB Classifier
### here the model has been fitted with XTrain and YTrain, then it predicts with XTest
## for evaluating classifier we use accuracy score and confusion Matrix which are available by metrics of scikit learn
### GaussianNB Classifier has 0.72 accuracy
## MultinomialNB Classifier
### here again the model has been fitted then it predicts with XTest
### the accuracy of this Classifier is approximately 0.60
## KNN Classifier
### this model has 0.68 accuracy
## DecisionTreeClassifier
###  this model has 0.74 accuracy 
### in line 55 I used predict_proba to see the probability of both positive Diabetes and negitive diabetes for each patient
## in line I used graphviz for picturing the tree
## neural network (MLPClassifier)
### this classifier has 0.68
##svc 
### this classifiers has  0.71
## logestic regression classifier
### with 0.97 accuracy score is the best one in this occasion 

