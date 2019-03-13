def accuracy(crossTab):
    tot_Postive = crossTab[0][0] + crossTab[1][1]
    tot_predictions = crossTab[0][0] + crossTab[1][1] + crossTab[1][0] + crossTab[0][1]
    return ((tot_Postive / tot_predictions)) * 100



import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
wrkdir = os.chdir("E:\Study\Projects\Data Science With Python\Loan")
le = preprocessing.LabelEncoder()


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

summary_stats = train.describe()
missing = train.isna().any()
train['Credit_History'].unique()
train['Credit_History'].value_counts()

train.info

train['Gender'] = np.where(train['Gender'].isnull(),train['Gender'].mode(),train['Gender'])
train['Married'] = np.where(train['Married'].isnull(),train['Married'].mode(),train['Married'])
train['Dependents'] = np.where(train['Dependents'].isnull(),train['Dependents'].mode(),train['Dependents'])
train['Dependents'] = np.where(train['Dependents'] == '3+',5,train['Dependents'])
train['Self_Employed'] = np.where(train['Self_Employed'].isnull(),train['Self_Employed'].mode(),train['Self_Employed'])
train['LoanAmount'] = np.where(train['LoanAmount'].isnull(),train['LoanAmount'].mean(),train['LoanAmount'])
train['Loan_Amount_Term'] = np.where(train['Loan_Amount_Term'].isnull(),train['Loan_Amount_Term'].mean(),train['Loan_Amount_Term'])
train['Credit_History'] = np.where(train['Credit_History'].isnull(),train['Credit_History'].mode(),train['Credit_History'])


train['Gender'] = le.fit_transform(train['Gender'])
train['Married'] = le.fit_transform(train['Married'])
train['Education'] = le.fit_transform(train['Education'])
train['Self_Employed'] = le.fit_transform(train['Self_Employed'])
train['Property_Area'] = le.fit_transform(train['Property_Area'])
train['Loan_Status'] = le.fit_transform(train['Loan_Status'])

X = train[train.columns[1:len(train.columns)-1]]
Y = train[train.columns[len(train.columns)-1]]

#Logistic Regression
from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(X,Y)
prediction_lr = lr.predict(X)
cmatrix_lr = pd.crosstab(prediction_lr,Y)
print(accuracy(cmatrix_lr)) #   81.10749185667753

#Random Forest
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X,Y)
prediction_rf = rf.predict(X)
cmatrix_rf = pd.crosstab(prediction_rf,Y)
print(accuracy(cmatrix_rf)) #   99.0228013029316


#Predictng using NaiveBasian 
from sklearn import naive_bayes
nb = naive_bayes.GaussianNB()
nb.fit(X,Y)
predict_nb = nb.predict(X)
cmatrix_nb = pd.crosstab(predict_nb,Y)
print(accuracy(cmatrix_nb)) #   79.80456026058633

#Predctng usng KNN
from sklearn import neighbors
KN = neighbors.KNeighborsClassifier()
KN.fit(X,Y)
predictKN = KN.predict(X)
cmatrix_KN = pd.crosstab(predictKN, Y)
print(accuracy(cmatrix_KN)) #   72.9641693811075

#Predcting usng SVM
from sklearn import svm
SVM = svm.SVC()
SVM.fit(X,Y)
predictSVM = SVM.predict(X)
cmatrix_SVM = pd.crosstab(predictSVM, Y)
print(accuracy(cmatrix_SVM)) #   100.0

#Predicting using MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))   
mlp.fit(X,Y)
Predict_MLP =mlp.predict(X)
cmatrix_MLP = pd.crosstab(Predict_MLP,Y)
print(accuracy(cmatrix_MLP)) #   43.97394136807817

#Dividing the train data into 2 parts so that one part will be consdered as trainData and Other wll be considered as test Data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33, random_state=42)

#Logistic Regression
from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(X_train,Y_train)
prediction_lr = lr.predict(X_test)
cmatrix_lr = pd.crosstab(prediction_lr,Y_test)
print(accuracy(cmatrix_lr)) #   81.10749185667753 79.80295566502463


#Random Forest
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train,Y_train)
prediction_rf = rf.predict(X_test)
cmatrix_rf = pd.crosstab(prediction_rf,Y_test)
print(accuracy(cmatrix_rf)) #   99.0228013029316 77.33990147783251


#Predictng using NaiveBasian 
from sklearn import naive_bayes
nb = naive_bayes.GaussianNB()
nb.fit(X_train,Y_train)
predict_nb = nb.predict(X_test)
cmatrix_nb = pd.crosstab(predict_nb,Y_test)
print(accuracy(cmatrix_nb)) #   79.80456026058633 79.80295566502463

#Predctng usng KNN
from sklearn import neighbors
KN = neighbors.KNeighborsClassifier()
KN.fit(X_train,Y_train)
predictKN = KN.predict(X_test)
cmatrix_KN = pd.crosstab(predictKN, Y_test)
print(accuracy(cmatrix_KN)) #   72.9641693811075 57.14285714285714

#Predcting usng SVM
from sklearn import svm
SVM = svm.SVC()
SVM.fit(X_train,Y_train)
predictSVM = SVM.predict(X_test)
cmatrix_SVM = pd.crosstab(predictSVM, Y_test)
print(accuracy(cmatrix_SVM)) #   100.0 65.02463054187191

#Predicting using MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))   
mlp.fit(X_train,Y_train)
Predict_MLP =mlp.predict(X_test)
cmatrix_MLP = pd.crosstab(Predict_MLP,Y_test)
print(accuracy(cmatrix_MLP)) #   43.97394136807817 64.5320197044335

#-----------------------------------test Data Starts--------------------------------
test['Gender'] = np.where(test['Gender'].isnull(),test['Gender'].mode(),test['Gender'])
test['Married'] = np.where(test['Married'].isnull(),test['Married'].mode(),test['Married'])
test['Dependents'] = np.where(test['Dependents'].isnull(),test['Dependents'].mode(),test['Dependents'])
test['Dependents'] = np.where(test['Dependents'] == '3+',5,test['Dependents'])
test['Self_Employed'] = np.where(test['Self_Employed'].isnull(),test['Self_Employed'].mode(),test['Self_Employed'])
test['LoanAmount'] = np.where(test['LoanAmount'].isnull(),test['LoanAmount'].mean(),test['LoanAmount'])
test['Loan_Amount_Term'] = np.where(test['Loan_Amount_Term'].isnull(),test['Loan_Amount_Term'].mean(),test['Loan_Amount_Term'])
test['Credit_History'] = np.where(test['Credit_History'].isnull(),test['Credit_History'].mode(),test['Credit_History'])


test['Gender'] = le.fit_transform(test['Gender'])
test['Married'] = le.fit_transform(test['Married'])
test['Education'] = le.fit_transform(test['Education'])
test['Self_Employed'] = le.fit_transform(test['Self_Employed'])
test['Property_Area'] = le.fit_transform(test['Property_Area'])


test.dtypes
prediction_lr = lr.predict(test[test.columns[1:len(test.columns)]])
predict_nb = nb.predict(test[test.columns[1:len(test.columns)]])
predict_rf = rf.predict(test[test.columns[1:len(test.columns)]])

#predictSVM.replace({ 0 : 'N', 1: 'Y'}, inplace=True)


submission1 = pd.DataFrame({'Loan_ID': test['Loan_ID'], 'Loan_Status': prediction_lr})

submission1.to_csv("lr_submission.csv", sep=',', index = False)

