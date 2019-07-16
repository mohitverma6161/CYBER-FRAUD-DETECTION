import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
dataset1 = pd.read_csv('CreditCardFraud.csv')
dataset = pd.read_csv('CreditCardFraud2.csv')  
dataset.head()
X = dataset.drop('isFraud', axis=1)  
y = dataset['isFraud']
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report,  accuracy_score
  
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

plt.plot(dataset1['nameOrig'],dataset1['amount']) 
plt.show()

demo=[[1,3,181.00,8400,181.0,0.00,38997010,21182.0,0.0],[2,2,11685,123003654,105555,168988.02,1978894566,0,0]]
y=classifier.predict(demo)
int(round(y[0],0))
