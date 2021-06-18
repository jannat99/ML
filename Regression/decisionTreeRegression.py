#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#importing dataset
dataset = pd.read_csv('polynomial&decisionTree.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

#training decision tree regression model on dataset
r = DecisionTreeRegressor(random_state=0)
r.fit(X,Y)

#prediction for x=6.5
dp = r.predict([[6.5]])
print("\nSalary for level 6.5 : ",dp)

#visualizing decision tree model for high resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, Y, color='red')
plt.plot(X_grid, r.predict(X_grid), color='blue')
plt.title('Salary vs Level (Decision Tree Regression)')
plt.xlabel('Level') 
plt.ylabel('Salary')
plt.show()