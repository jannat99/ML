#importing libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values
print("\nDataset without last column : \n",X)
print("\nDataset of only last column : \n",Y)

#taking care of missing data
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\nAfter getting missing data : \n",X)

#encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print("\nAfter one hot encoding : \n",X)

#encoding dependent variable
le = LabelEncoder()
Y = le.fit_transform(Y)
print("\nAfter label encoding : \n",Y)

#splitting dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
print("\nDataset of Train-X : \n",X_train)
print("\nDataset of Test-X : \n",X_test)
print("\nDataset of Train-Y : \n",Y_train)
print("\nDataset of Test-Y : \n",Y_test)

#feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("\nAfter Feature Scaling the trained data of X : \n",X_train)
print("\nAfter Feature Scaling the test data of X : \n",X_test)

#data input
input = np.array([[5,2,6],[3,1,9],[6,6,2]])
print("\nInputs are: \n",input)

#Data Binarize
binarize = preprocessing.Binarizer(threshold=4).transform(input)
print("\nBinarized Data : \n",binarize)

#Data Scaling
scale = preprocessing.MinMaxScaler(feature_range=(3,6))
scaled = scale.fit_transform(input)
print("\nScaled Data : \n",scaled)

#Data Normalize
l1 = preprocessing.normalize(input, norm='l1')
l2 = preprocessing.normalize(input, norm='l2')
print("\nl1 Normalized Data : \n",l1)
print("\nl2 Normalized Data : \n",l2)