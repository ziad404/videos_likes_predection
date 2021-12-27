import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm,tree
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
import os
import joblib
import time


def labelEncoding(colName,dataFrame):
    le = LabelEncoder()
    return le.fit_transform(dataFrame[colName])


def processMissingValues(dataFrame,colName, val):
    return dataFrame[colName].fillna(value=val)


def feature_scaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X


df = pd.read_csv('VideoLikesDatasetClassification.csv')
df.head()
df["video_description"] = processMissingValues(df,"video_description","No description")
df["video_description"] = labelEncoding("video_description",df)
df["comments_disabled"] = labelEncoding("comments_disabled",df)
df["ratings_disabled"] = labelEncoding("ratings_disabled",df)
df["video_error_or_removed"] = labelEncoding("video_error_or_removed",df)
df["video_id"] = labelEncoding("video_id",df)
df["trending_date"] = labelEncoding("trending_date", df)
df["tags"] = labelEncoding("tags",df)
df["title"] = labelEncoding("title",df)
df["channel_title"] = labelEncoding("channel_title",df)
df["publish_time"] = labelEncoding("publish_time",df)
df["VideoPopularity"] = labelEncoding("VideoPopularity",df)
df.head()
X = df.drop("VideoPopularity",1)
Y = df[['VideoPopularity']].iloc[:,:]
Y = np.reshape(Y,-1)
lreg = LinearRegression()
sfs = SFS(lreg,k_features=3)
sfs.fit(X,Y)
feat_names = sfs.k_feature_names_
x = df[['publish_time', 'views', 'video_description']]
x = feature_scaling(x,0,50)
z=pd.DataFrame(x)
x = np.array(x)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(x, Y, test_size=0.2,shuffle=False)
if os.path.exists('./model5.sav'):
    model = joblib.load('model5.sav')
else:
    model = tree.DecisionTreeClassifier()
    history = model.fit(X_train, y_train)
    filename = './model5.sav'
    joblib.dump(model, filename)

y_pred = model.predict(X_test)
acc = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        acc+=1
acc = acc/len(y_test)*100
print('accuracy: '+ f'{acc:.9f}' )