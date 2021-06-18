#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#importing dataset
dataset = pd.read_csv('polynomial&decisionTree.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

#train linear regression model on dataset
r1 = LinearRegression()
r1.fit(X,Y)

#train polynomial regression model on dataset
p = PolynomialFeatures(degree=5)
xp = p.fit_transform(X)
r2 = LinearRegression()
r2.fit(xp, Y)

#visualizing linear regression result
plt.scatter(X, Y, color='red')
plt.plot(X, r1.predict(X), color='blue')
plt.title('Salary vs Level (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#visualizing polynomial regression result
plt.scatter(X, Y, color='red')
plt.plot(X, r2.predict(p.fit_transform(X)), color='blue')
plt.title('Salary vs Level (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#visualizing polynomial regression result for smooth curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, r2.predict(p.fit_transform(X_grid)), color='blue')
plt.title('Salary vs Level (Polynomial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#linear regression prediction for example x=6.5
lp = r1.predict([[6.5]])
print("\nSalary for level 6.5 : ",lp)

#polynomial regression prediction for example x=6.5
pp = r2.predict(p.fit_transform([[6.5]]))
print("\nSalary for level 6.5 : ",pp)