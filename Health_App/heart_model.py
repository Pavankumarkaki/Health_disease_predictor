import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
import pickle
def read():
    data=pd.read_csv("D:/Datasets/heart.csv")
    print("Some information about data :")
    print("shape of the data is :",data.shape)
    print('Columns in the data are:\n',data.columns)
    print('the first 5 rows in the data :\n',data.head())
    return data

# cleaning data
def  preprocess():
    print("sum of null values of each column\n",data.isnull().sum())
    double_entries=data.duplicated().sum()
    print('counting of double entries of data is ',double_entries)
    if double_entries !=0:
        data.drop_duplicates().sum()
        print('Shape of the data after droping dupicates',data.shape)

def Encoding():
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    pickle.dump(le,open('LabelEncoding.sav','wb'))




# standardising and spliting data for training and testing
def feature_Extraction(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
    return x_train,y_train,x_test,y_test

# creating model
def create_LRmodel():
    LR_model=LogisticRegression()
    LR_model.fit(x_train,y_train)
    return LR_model


if __name__=='__main__':
    data=read()
    preprocess()
    x = data.drop('output', axis=1)
    y = data['output']
    Encoding()
    x_train,y_train,x_test,y_test=feature_Extraction(x,y)
    print("Shape of training and test data ",x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    model=create_LRmodel()
    print('Accuracy score of LR Model test is ',accuracy_score(y_test,model.predict(x_test))*100)
    print('Accuracy score of LR Model train  is ', accuracy_score(y_train, model.predict(x_train)) * 100)
    pickle.dump(model,open('trained_LR_model_for_heart.sav','wb'))



