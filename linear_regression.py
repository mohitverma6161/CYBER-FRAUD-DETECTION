import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
dataset= pd.read_csv('CreditCardFraud2.csv')  
dataset.head()
Xs = dataset.drop('isFraud', axis=1)  
y = dataset['isFraud']
reg = LinearRegression()
reg.fit(Xs, y)
#X = np.column_stack((dataset['step'], dataset['type'], dataset['amount'], dataset['nameOrig'], dataset['oldbalaneOrg'], dataset['newbalanceOrig'],dataset['nameDest'], dataset['oldbalaneDest'], dataset['newbalanceDest']))
X = dataset.drop('isFraud', axis=1)  
y = dataset['isFraud']
#X2 = sm.add_constant(X)
#est = sm.OLS(y, X2)
#est2 = est.fit()
#print(est2.summary())
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)

classifier = LinearRegression()  
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred.round()))  
print(classification_report(y_test, y_pred.round()))
print(accuracy_score(y_test, y_pred.round()))
