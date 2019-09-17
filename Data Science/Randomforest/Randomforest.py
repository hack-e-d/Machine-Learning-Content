import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math as math
import numpy as np 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

balance_data = pd.read_csv('Boston_Housing_Data.csv') 
# Printing the dataswet shape 
print ("Dataset Length: ", len(balance_data)) 
print ("Dataset Shape: ", balance_data.shape) 
	
# Printing the dataset obseravtions 
print ("Dataset: ",balance_data.head()) 
X = balance_data.values[:, 0:12] 
Y = balance_data.values[:, 13] 

# Spliting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 

#Creating a gini based model
model = RandomForestRegressor(n_estimators = 500, min_samples_split = 40, max_features= "auto") 

# Actual training process 
model.fit(X_train, y_train) 

# Predicton on testdata 
y_pred = model.predict(X_test) 
print("Predicted values:") 
print(y_pred)

print("Accuracy : ",model.score(X_test,y_test))

res = y_test-y_pred
print("Residue :",res)

res_sq = res**2
mse=res_sq.mean()
print("Mean Residue :",mse)

rmse= math.sqrt(mse)
print("Root Mean Residue :",rmse)
