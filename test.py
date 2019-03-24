import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
#print (data)
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(X_train.head())
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)
y_pred = clf.predict(X_test)
#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print x[i]
    
#printing first five expectations
print("\nThe expectation:\n")
print y_test.head()