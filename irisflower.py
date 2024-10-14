import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
column=['sepal length','sepal width','petal length','petal width','species']
iris= pd.read_csv('IRIS.csv')
iris.head(150)
print(iris.columns)
sns.pairplot(iris, hue='species')
data=iris.values
x=data[:,0:4]
y=data[:,4]
print(y)
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)
print(y_train)
from sklearn.svm import SVC
model_svc=SVC()
model_svc.fit(x_train,y_train)
prediction1=model_svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction1)*100)
from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(x_train,y_train)
prediction2=model_LR.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction2)*100)
from sklearn.tree import DecisionTreeClassifier
model_DTC=DecisionTreeClassifier()
model_DTC.fit(x_train,y_train)
prediction3=model_DTC.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction3)*100)
X_new=np.array([[1,2,1,1.1],[8,2.2,7,9],[5.3,2.5,4.6,4]])
prediction_species=model_DTC.predict(X_new)
print("prediction of species: {}".format(prediction_species))
