#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

#importing dataset
dataset = pd.read_csv('multipleLinearRegression.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#training multiple linear regression model on the training set
r = LinearRegression()
r.fit(X_train, Y_train)

#predict test result
Y_predict = r.predict(X_test)
np.set_printoptions(precision=2)
print("\nY-Prediction    Y-Test\n",np.concatenate((Y_predict.reshape(len(Y_predict),1),Y_test.reshape(len(Y_test),1)),1))

#single prediction for example the profit with R&D=160000, Administration=130000, Marketing=300000, State='California')
sp = r.predict([[1, 0, 0, 160000, 130000, 300000]])
print("\nProfit for specified example : ",sp)

#predict b0,b1,b2,b3,b4,b5,b6 for y= b0+ b1x1+ b2x2+ b3x3+ b4x4+ b5x5+ b6x6
i = r.intercept_
c = r.coef_
print("\nThe intercept(b0) is : ",i)
print("\nThe co-efficient[b1 b2 b3 b4 b5 b6] is : ",c)






