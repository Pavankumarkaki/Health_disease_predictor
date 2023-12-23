import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC


# Reading and understanding the data
def read():
    data=pd.read_csv("D:/Datasets/diabetes.csv")
    print("Some information about data :")
    print("shape of the data is :",data.shape)
    print('Columns in the data are:\n',data.columns)
    print('the first 5 rows in the data :\n',data.head())
    return data

# cleaning data
def  preprocess():
    print("sum of null values of each column\n",data.isnull().sum())
    print('counting of double entries of data is ',data.duplicated().sum())



# standardising and spliting data for training and testing
def feature_Extraction(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
    return x_train,y_train,x_test,y_test

# creating model
def create_LRmodel():
    from sklearn.linear_model import LogisticRegression
    LR_model=LogisticRegression()
    LR_model.fit(x_train,y_train)
    return LR_model


if __name__=='__main__':
    data=read()
    preprocess()
    x = data.drop('Outcome', axis=1)
    y = data['Outcome']
    x_train,y_train,x_test,y_test=feature_Extraction(x,y)
    print("Shape of training and test data ",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model=create_LRmodel()
    print('Accuracy score of LR Model is ',accuracy_score(y_test,model.predict(x_test))*100)
    pickle.dump(model,open('trained_model_diabetes.sav','wb'))



    # random forest model creation
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    # predictions
    print('Accuracy score of RandomForest classifier with train data', accuracy_score(rfc.predict(x_train), y_train))
    print('Accuracy score of RandomForestClassifier with test data', accuracy_score(rfc.predict(x_test), y_test))


    # Suppoort vector machine

    svm_classifier=svm.SVC()
    svm_classifier.fit(x_train,y_train)
    # predictions
    print('Accuracy score of Support vector classifier with train data', accuracy_score(y_train,svm_classifier.predict(x_train)))
    print('Accuracy score of Support vector Classifier with test data', accuracy_score( y_test,svm_classifier.predict(x_test)))


