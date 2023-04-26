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
