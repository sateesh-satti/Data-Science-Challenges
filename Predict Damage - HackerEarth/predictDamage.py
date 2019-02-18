import  pandas as pd
import os

os.chdir(r"C:\Users\sateesh\PycharmProjects\Learning\PredictDamage")
os.getcwd()
train = pd.read_csv("train.csv")
ownerShip = pd.read_csv("Building_Ownership_Use.csv")
structure = pd.read_csv("Building_Structure.csv")
train_ownership = pd.merge(train, ownerShip, on='building_id', how='inner')
train_ownership_str = pd.merge(train_ownership, structure, on='building_id', how='inner')
train_ownership_str.shape
X = train_ownership_str[train_ownership_str.columns.difference(['bulding_id','damage_grade'])]
Y = pd.DataFrame(train_ownership_str.loc[:, 'damage_grade'])
colsToExclude = ('building_id')
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(2)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns

missing_values_table(X)

def imputeMissingValues(df,categoryMethod,numericalMethod,categoryPackage,numericalPacakge,colsToExclude):
    catModule = __import__(categoryPackage)
    catFunc = getattr(catModule, categoryMethod)
    numModule = __import__(numericalPacakge)
    numFunc = getattr(numModule, numericalMethod)
    whereModule = __import__("numpy")
    whereFunc = getattr(whereModule, "where")
    for col in df:
      if df[col].dtype == 'object' and col not in colsToExclude:
          df[col] = whereFunc(df[col].isnull(),catFunc(df[col]),df[col])
      elif df[col].dtype in ('float64', 'int64', 'int32' ,'float32' ) and col not in colsToExclude:
          df[col] = whereFunc(df[col].isnull(),numFunc(df[col]),df[col])

imputeMissingValues(X,"mode","nanmean","statistics","numpy",colsToExclude)

X.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
Y.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

def labelEncodingOfVariables(df, minNoOfUniqueClass, colsToExclude):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    noOfColsLabeld = 0
    colsLabelled = list()
    for col in df:
        if df[col].dtype == 'object' and col not in colsToExclude:
            if len(list(df[col].unique())) <= minNoOfUniqueClass:
                le.fit(df[col])
                colsLabelled.append(col)
                df[col] = le.transform(df[col])
                noOfColsLabeld += 1
                print("Label Encoding done for %s" % col)
    print("Total Number of coloumns Labelled are %d" % noOfColsLabeld)
    return colsLabelled


colsLabelled_X = labelEncodingOfVariables(X, 3, colsToExclude)
colsLabelled_Y = labelEncodingOfVariables(Y, 5, colsToExclude)

def oneHotEncodingOfVariable(data,colsToExclude):
    import pandas as pd
    for col in data:
      if data[col].dtype == 'object' and col not in colsToExclude:
          data = pd.get_dummies(data, columns= [col], prefix=[col], prefix_sep='_')
    return data

X = oneHotEncodingOfVariable(X,('building_id'))
X.to_pickle("./dummy.pkl")

def removeConstantVariables(data,colsToExclude):
    colsRemoved = list()
    for col in data:
      if col not in colsToExclude and data[col].std() == 0:
        colsRemoved.append(col)
        data.drop(col, axis=1, inplace = True)
    return colsRemoved

colsRemoved = removeConstantVariables(X,colsToExclude)


def removeDuplicateVariables(data):
    import numpy as np
    cols = data.columns
    colsScanned = list()
    colsToRemove = list()
    for i in range(len(cols) - 1):
        values = data[cols[i]].values
        dupColoumns = list()
        for j in range(i + 1, len(cols)):
            if np.array_equal(values, data[cols[j]].values):
                colsToRemove.append(cols[j])
                if cols[j] not in colsScanned:
                    dupColoumns.append(cols[j])
                    colsScanned.append(cols[j])
    return colsToRemove


colsToRemove = removeDuplicateVariables(X)
X.drop(colsToRemove,axis=1,inplace=True)

def correlation(dataset, threshold):
    col_corr = {} # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr[colname] = corr_matrix.iloc[i, j]
                if colname in dataset.columns:
                   del dataset[colname] # deleting the column from the dataset
    for key, val in col_corr.items():
        print(key, "=>", val)
    # print(dataset)

correlation(X,0.55)

def rmHighlyCorrVars(data,percentageToRemove):
    correlatedVars = set()
    corrMatrix = data.corr()
    for i in range(len(corrMatrix)):
      for j in range(i):
          if corrMatrix.iloc[i,j] >= percentageToRemove and corrMatrix.columns[i] not in colsToRemove:
            correlatedVars.add(corrMatrix.columns[i])
    return correlatedVars

correlatedVarsByCorr =  rmHighlyCorrVars(X,0.65)

X.drop('building_id',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.33)

## List of Models Evaluating
def evaluateClassificationModels(X_train,X_test,Y_train,Y_test):
    #Imports
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    from sklearn import ensemble
    from sklearn import neighbors
    from sklearn.neural_network import MLPClassifier
    from sklearn import tree
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_score,recall_score
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier
    #Models
    classificationModels = list()
    keysOfParam = set()
    params_lr = [
    {'penalty': ['l1'],'C': [1, 10, 100, 1000], 'solver' : ['liblinear','saga']},
    {'penalty': ['l2'],'C': [1, 10, 100, 1000], 'solver' : ['newton-cg','sag','lbfgs']}
    ]
    lr = linear_model.LogisticRegression()
    classificationModels.append(('Logistic Regression',lr,params_lr))
    params_rf = [
    {'n_estimators' : [10,20,30,50,100],'criterion' : ['gini'],'max_features':['sqrt','log2']},
    {'n_estimators' : [10,20,30,50,100],'criterion' : ['entropy'],'max_features':['sqrt','log2']}
     ]
    rf = ensemble.RandomForestClassifier()
    classificationModels.append(('RandomForest',rf,params_rf))
    params_kn = [
       {'n_neighbors' : [1,5,10,15,20,25,30],'weights' : ['uniform'],'algorithm' : ['auto','ball_tree','kd_tree','brute'],'leaf_size' : [5,10,15,20,25,30],'p' : [1,2]},
       {'n_neighbors' : [1,5,10,15,20,25,30],'weights' : ['distance'],'algorithm' : ['auto','ball_tree','kd_tree','brute'],'leaf_size' : [5,10,15,20,25,30],'p' : [1,2]}
     ]
    kn = neighbors.KNeighborsClassifier()
    classificationModels.append(('KNeighbors',kn,params_kn))
    params_mlp = [
       {'hidden_layer_sizes' : [50,75,100,125,150,175,200],'activation' : ['identity','logistic','tanh','relu'],'solver' : ['lbfgs','sgd','adam'],'learning_rate' : ['constant','invscaling','adaptive']},
       {'hidden_layer_sizes' : [50,75,100,125,150,175,200],'activation' : ['identity','logistic','tanh','relu'],'solver' : ['sgd'],'learning_rate' : ['constant','invscaling','adaptive']}
     ]
    mlp = MLPClassifier()
    classificationModels.append(('MLPClassifier',mlp,params_mlp))

    params_tr = [
       {'criterion' : ['gini'],'splitter' : ['best','random'],'max_features' : ['auto','log2']},
       {'criterion' : ['entropy'] ,'splitter' : ['best','random'],'max_features' : ['auto','log2']}
     ]
    tr = tree.DecisionTreeClassifier()
    classificationModels.append(('DecisionTree',tr,params_tr))

    params_boost = [
       {'n_estimators' : [10,20,30,50,100],'random_state' : [5,7,8,9,11]},
       {'n_estimators' : [60,70,80,90,110],'random_state' : [12,13,14]}
     ]
    ada = AdaBoostClassifier()
    classificationModels.append(('AdaBoostClassifier',ada,params_boost))
    params_SGD = [
         {'loss' : ['hinge','log','modified_huber','squared_hinge','perceptron'], 'penalty' : ['l1']},
         {'loss' : ['hinge','log','modified_huber','squared_hinge','perceptron'], 'penalty' : ['l2']},
         {'loss' : ['hinge','log','modified_huber','squared_hinge','perceptron'], 'penalty' : ['elasticnet']}
       ]
    SGD = SGDClassifier()
    classificationModels.append(('Stochastic Gradient Classifier',SGD,params_SGD))
    params_PA = [
         {'C': [1, 10, 100, 1000],'random_state' : [5,7,8,9,11]}
       ]
    PA = PassiveAggressiveClassifier()
    classificationModels.append(('Passive Aggressive Classifier',PA,params_PA))
    #     params_SVC = [
    #             {'C': [1, 10,],'kernel' : ['rbf'],'degree' : [3]}
    # #          {'C': [1, 10, 100, 1000],'kernel' : ['linear','rbf','poly','sigmoid'],'degree' : [3,4,5,6]}
    #         ]
    #     svc = SVC()
    #     classificationModels.append(('Support Vector Classifier',svc,params_SVC))
    for name,model,params in classificationModels :
        print("------------------Running model -- {}  -------------------".format(name))
        for i in params:
          for j in i.keys():
            keysOfParam.add(j)
        grid = GridSearchCV(model, params, n_jobs=-1,verbose=1, scoring='accuracy', cv=3)
        grid.fit(X_train, Y_train.values.ravel())
        best_parameters = grid.best_estimator_.get_params()
        print("------------------Best Parameters for the model -- {}  -------------------".format(name))
        for param_name in sorted(keysOfParam):
            if param_name in best_parameters.keys():
                print('%s: %r' % (param_name, best_parameters[param_name]))
        predictions = grid.predict(X_test)
        print('####Accuracy:', accuracy_score(Y_test, predictions))
        print('####Precision:', precision_score(Y_test, predictions))
        print('####Recall:', recall_score(Y_test, predictions))
        print("------------------Model came to END -- {}  -------------------".format(name))


evaluateClassificationModels(X_train,X_test,Y_train,Y_test)

import  numpy as np
y_preds_test = pd.DataFrame()
y_preds_test = pd.read_csv("preds_classes_output (1).csv")
y_preds_test.drop("Unnamed: 0",axis = 1,inplace=True)
y_preds_test['damage_grade'] = np.where(y_preds_test['damage_grade'] == 0,'Grade 1',y_preds_test['damage_grade'])
y_preds_test['damage_grade'] = np.where(y_preds_test['damage_grade'] == '1','Grade 2',y_preds_test['damage_grade'])
y_preds_test['damage_grade'] = np.where(y_preds_test['damage_grade'] == '2','Grade 3',y_preds_test['damage_grade'])
y_preds_test['damage_grade'] = np.where(y_preds_test['damage_grade'] == '3','Grade 4',y_preds_test['damage_grade'])
y_preds_test['damage_grade'] = np.where(y_preds_test['damage_grade'] == '4','Grade 5',y_preds_test['damage_grade'])

y_preds_test.to_csv("prediction_rf.csv",index=False)

#from sklearn.externals import joblib
# save the model to disk
# filename = 'finalized_model.sav'
# joblib.dump(model, filename)
#
# # some time later...
#
# # load the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, Y_test)
# print(result)