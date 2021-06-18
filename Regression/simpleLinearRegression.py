#importing libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#importing dataset
dataset = pd.read_csv('simpleLinearRegression.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#training simple linear regression model on the training set
r = LinearRegression()
r.fit(X_train, Y_train)

#predict test result
Y_predict = r.predict(X_test)

#visualizing training result
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, r.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing test result
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, r.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#single prediction for example x=12
sp = r.predict([[12]])
print("\nSalary for 12 years of experience : ",sp)

#predict b0,b1 for y=b0+b1x
i = r.intercept_
c = r.coef_
print("\nThe intercept(b0) is : ",i)
print("\nThe co-efficient(b1) is : ",c)

