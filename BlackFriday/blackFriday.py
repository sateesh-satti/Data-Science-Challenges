import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import os
wrkdir = os.chdir("E:\Study\Projects\Data Science With Python\Black Frday From AV")
le = preprocessing.LabelEncoder()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

summary_stats = train.describe()
missing = train.isna().any()

train.columns


#Treating Missing Values
train['Product_Category_2'] = np.where(train['Product_Category_2'].isnull(),train['Product_Category_2'].mean(),train['Product_Category_2'])
train['Product_Category_3'] = np.where(train['Product_Category_3'].isnull(),train['Product_Category_3'].mean(),train['Product_Category_3'])

#Convertng labels to integers
train['Gender'] = le.fit_transform(train['Gender'])

#Treatment of AGe Intervals
train['Age'].value_counts()
train['Age'] = np.where(train['Age'] == '0-17',(0+17)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '51-55',(51+55)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '46-50',(46+50)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '18-25',(18+25)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '36-45',(36+45)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '26-35',(26+35)/2,train['Age'])
train['Age'] = np.where(train['Age'] == '55+',56,train['Age'])

#Converting lebesl for City Category
train['City_Category'].value_counts()
train['City_Category'] = le.fit_transform(train['City_Category'])
#Converting lebesl for Product ID
train['Product_ID'].value_counts()
train['Product_ID'] = le.fit_transform(train['Product_ID'])



#Changing the values for 4+ to 5
train['Stay_In_Current_City_Years'].value_counts()
train['Stay_In_Current_City_Years'] = np.where(train['Stay_In_Current_City_Years'] == '4+',5,train['Stay_In_Current_City_Years'])

#Seperating the Independant and dependant Varables
X = train[train.columns[1:len(train.columns)-1]]
Y = train[train.columns[len(train.columns)-1]]

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, Y)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

y_train_pred = regressor.predict(X)

np.sqrt(mean_squared_error(Y,y_train_pred))



#-------------------------------Test------------------------------------

test.isna().any()



#Treating Missing Values
test['Product_Category_2'] = np.where(test['Product_Category_2'].isnull(),test['Product_Category_2'].mean(),test['Product_Category_2'])
test['Product_Category_3'] = np.where(test['Product_Category_3'].isnull(),test['Product_Category_3'].mean(),test['Product_Category_3'])

#Convertng labels to integers
test['Gender'].value_counts()
test['Gender'] = le.fit_transform(test['Gender'])

#Treatment of AGe Intervals
test['Age'].value_counts()
test['Age'] = np.where(test['Age'] == '0-17',(0+17)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '51-55',(51+55)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '46-50',(46+50)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '18-25',(18+25)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '36-45',(36+45)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '26-35',(26+35)/2,test['Age'])
test['Age'] = np.where(test['Age'] == '55+',56,test['Age'])

#Converting lebesl for City Category
test['City_Category'].value_counts()
test['City_Category'] = le.fit_transform(test['City_Category'])
#Converting lebesl for Product ID
test['Product_ID'].value_counts()
test['Product_ID'] = le.fit_transform(test['Product_ID'])



#Changing the values for 4+ to 5
test['Stay_In_Current_City_Years'].value_counts()
test['Stay_In_Current_City_Years'] = np.where(test['Stay_In_Current_City_Years'] == '4+',5,test['Stay_In_Current_City_Years'])

#Predicting the Test Data

test.head()

X_test = test[test.columns[1:len(test.columns)]]

y_pred = regressor.predict(X_test)

submission1 = pd.DataFrame({'User_ID': test['User_ID'],'Product_ID': test['Product_ID'], 'Purchase': y_pred})
submission1 = submission1[['User_ID', 'Product_ID' ,'Purchase']]

submission1.to_csv("rf_submission.csv", sep=',', index = False)


