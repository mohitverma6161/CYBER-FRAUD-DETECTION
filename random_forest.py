import pandas as pd  
import numpy as np 
dataset = pd.read_csv('CreditCardFraud2.csv')  
dataset.head()
X = dataset.iloc[:, 0:9].values  
y = dataset.iloc[:, 9].values  
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cutoff = 0.99                        
y_pred_classes = np.zeros_like(y_pred)    
y_pred_classes[y_pred > cutoff] = 1 

y_test_classes = np.zeros_like(y_pred)
y_test_classes[y_test_classes > cutoff] = 1

print(confusion_matrix(y_test_classes, y_pred_classes))
print(classification_report(y_test_classes, y_pred_classes)) 
print(accuracy_score(y_test_classes, y_pred_classes))

demo=[[1,1,9839.64,1231006815,170136,160296.36,1979787155,0,0],[2,2,11685,123003654,105555,168988.02,1978894566,0,0]]
y=regressor.predict(demo)
int(round(y[0],0))
